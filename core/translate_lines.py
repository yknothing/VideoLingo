from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_expressiveness
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
console = Console()

def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    
    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

    return {"status": "success", "message": "Translation completed"}

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index = 0):
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt)

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name):
        def valid_faith(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['direct'])
        def valid_express(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['free'])
        for retry in range(3):
            if step_name == 'faithfulness':
                result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid_faith, log_title=f'translate_{step_name}')
            elif step_name == 'expressiveness':
                result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid_express, log_title=f'translate_{step_name}')
            if len(lines.split('\n')) == len(result):
                return result
            if retry != 2:
                console.print(f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed, Retry...[/yellow]')
        raise ValueError(f'[red]❌ {step_name.capitalize()} translation of block {index} failed after 3 retries. Please check `output/gpt_log/error.json` for more details.[/red]')

    ## Step 1: Faithful to the Original Text
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(lines.split('\n')), 'faithfulness')

    for i in faith_result:
        faith_result[i]["direct"] = faith_result[i]["direct"].replace('\n', ' ')

    # If reflect_translate is False or not set, use faithful translation directly
    reflect_translate = load_key('reflect_translate')
    if not reflect_translate:
        # If reflect_translate is False or not set, use faithful translation directly
        translate_result = "\n".join([faith_result[i]["direct"].strip() for i in faith_result])
        
        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(faith_result):
            table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
            if i < len(faith_result) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        
        console.print(table)
        return translate_result, lines

    ## Step 2: Express Smoothly  
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, len(lines.split('\n')), 'expressiveness')

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)

    translate_result = "\n".join([express_result[i]["free"].replace('\n', ' ').strip() for i in express_result])

    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_expressiveness.json`[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

    return translate_result, lines

def translate_lines_batch(chunks_data, theme_prompt):
    """
    Batch translate multiple chunks in a single LLM call to reduce API requests
    chunks_data: list of dict with keys: chunk, previous_content, after_content, things_to_note, index
    """
    if not chunks_data:
        return []
    
    # ------------
    # Prepare batch input for LLM
    # ------------
    batch_input = {}
    all_lines = []
    chunk_line_mapping = {}  # Track which lines belong to which chunk
    
    line_counter = 1
    for chunk_data in chunks_data:
        chunk_lines = chunk_data['chunk'].split('\n')
        chunk_start = line_counter
        
        for line in chunk_lines:
            batch_input[str(line_counter)] = {
                "origin": line,
                "chunk_index": chunk_data['index'],
                "direct": f"direct translation {line_counter}."
            }
            all_lines.append(line)
            line_counter += 1
        
        chunk_line_mapping[chunk_data['index']] = {
            'start': chunk_start,
            'end': line_counter - 1,
            'lines': chunk_lines
        }
    
    # ------------
    # Generate shared context (use first chunk's context as primary)
    # ------------
    primary_chunk = chunks_data[0]
    shared_prompt = generate_shared_prompt(
        primary_chunk.get('previous_content'),
        primary_chunk.get('after_content'), 
        theme_prompt,
        primary_chunk.get('things_to_note')
    )
    
    # ------------
    # Batch faithfulness translation
    # ------------
    all_lines_text = '\n'.join(all_lines)
    
    def valid_batch_faith(response_data):
        return valid_translate_result(response_data, [str(i) for i in range(1, len(all_lines)+1)], ['direct'])
    
    def valid_batch_express(response_data):
        return valid_translate_result(response_data, [str(i) for i in range(1, len(all_lines)+1)], ['free'])
    
    # Faithfulness step
    prompt1 = get_prompt_faithfulness(all_lines_text, shared_prompt)
    faith_result = ask_gpt(prompt1, resp_type='json', valid_def=valid_batch_faith, log_title='translate_batch_faithfulness')
    
    for i in faith_result:
        faith_result[i]["direct"] = faith_result[i]["direct"].replace('\n', ' ')
    
    # Check if need expressiveness step
    reflect_translate = load_key('reflect_translate')
    if not reflect_translate:
        # Use faithful translation directly
        final_result = faith_result
    else:
        # Expressiveness step
        prompt2 = get_prompt_expressiveness(faith_result, all_lines_text, shared_prompt)
        express_result = ask_gpt(prompt2, resp_type='json', valid_def=valid_batch_express, log_title='translate_batch_expressiveness')
        final_result = express_result
    
    # ------------
    # Split results back to individual chunks
    # ------------
    chunk_results = {}
    for chunk_data in chunks_data:
        chunk_idx = chunk_data['index']
        mapping = chunk_line_mapping[chunk_idx]
        
        if reflect_translate:
            chunk_translation = '\n'.join([
                final_result[str(i)]["free"].replace('\n', ' ').strip() 
                for i in range(mapping['start'], mapping['end'] + 1)
            ])
        else:
            chunk_translation = '\n'.join([
                final_result[str(i)]["direct"].replace('\n', ' ').strip() 
                for i in range(mapping['start'], mapping['end'] + 1)
            ])
        
        chunk_results[chunk_idx] = {
            'translation': chunk_translation,
            'original': chunk_data['chunk']
        }
        
        # Display results for this chunk
        table = Table(title=f"Batch Translation Results - Chunk {chunk_idx}", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        
        for i in range(mapping['start'], mapping['end'] + 1):
            line_key = str(i)
            table.add_row(f"[cyan]Origin:  {final_result[line_key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {final_result[line_key]['direct']}[/magenta]")
            if reflect_translate and 'free' in final_result[line_key]:
                table.add_row(f"[green]Free:    {final_result[line_key]['free']}[/green]")
            if i < mapping['end']:
                table.add_row("[yellow]" + "-" * 30 + "[/yellow]")
        
        console.print(table)
    
    return chunk_results

if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)