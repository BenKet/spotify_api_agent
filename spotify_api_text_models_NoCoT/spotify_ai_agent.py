import gradio as gr
import requests
import json
import base64
import os
import re
from dotenv import load_dotenv
import copy  # Import the copy module

from transformers import AutoTokenizer

# We continue to use the Gemma tokenizer in a global variable:
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-27b-it", use_fast=True)
# For counting tokens with Qwen, a second tokenizer must be loaded:
tokenizer_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)

def extract_json_blocks(text: str):
    """
    Finds all JSON blocks from '{' to the corresponding '}'
    and returns them as a list.
    """
    # First, remove all code fences (`...`)
    text_no_fences = re.sub(r"`.*?`", "", text, flags=re.DOTALL)

    # Find all { ... } blocks (greedy or non-greedy):
    json_candidates = re.findall(r"\{.*?\}", text_no_fences, flags=re.DOTALL)

    return json_candidates

def safe_json_parse(model_output: str):
    """
    - Removes code fences.
    - Finds all JSON blocks (e.g., {...}).
    - Attempts to parse the first valid JSON block.
    - Returns the loaded Python dict or None on error.
    """
    candidates = extract_json_blocks(model_output)
    if not candidates:
        return None

    # Try to parse each found block
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            return parsed
        except json.JSONDecodeError:
            # If this block is not parsable, try the next one
            continue

    # If none of the blocks were parsable:
    return None

def load_env_variables():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory
    parent_dir = os.path.dirname(script_dir)
    # Construct the path to the .env file in the parent directory
    dotenv_path = os.path.join(parent_dir, '.env')

    if not load_dotenv(dotenv_path=dotenv_path): # Pass the path to the .env file
        raise ValueError(".env file could not be loaded. Ensure it exists and is correctly formatted in the parent directory.")
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    if not client_id or client_secret is None:
        raise ValueError("CLIENT_ID or CLIENT_SECRET not set.")
    return client_id, client_secret

def get_spotify_access_token():
    client_id, client_secret = load_env_variables()

    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = {
        'Authorization': f'Basic {base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()}'
    }
    auth_data = {
        'grant_type': 'client_credentials'
    }

    response = requests.post(auth_url, headers=auth_header, data=auth_data)

    if response.status_code != 200:
        raise Exception(f"Error fetching access token: {response.status_code} {response.text}")

    token_info = response.json()
    access_token = token_info.get('access_token')
    if not access_token:
        raise Exception("No access token received.")

    return access_token

# Load the prompt from a file
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Load the API documentation
def load_api_documentation():
    """
    Since we only use the text documentation as per requirement,
    we no longer need a selection. We directly load 'api_doc_text.txt'.
    """
    with open('api_doc_text.txt', 'r', encoding='utf-8') as f:
        return f.read()

# Function to count tokens (pure word tokenizing, for demo purposes)
def count_tokens(text):
    tokens = text.split()
    return len(tokens)

def generate_cot_and_extract_requests(question, model_api_url, model_parameters):
    """
    1) Queries the model with 'prompt'.
    2) Separates the output into 'Chain of Thought' (cot_context)
       and 'API Request Mapping' (via '**API Request Mapping:**').
    3) Searches for a JSON block. First for `json ... ` (Regex),
       then via safe_json_parse as fallback.
    """
    prompt = generate_model_input(question)
    model_output = call_model_api(prompt, model_api_url, model_parameters)
    print(model_output)

    # Split the model output at '**API Request Mapping:**'
    parts = model_output.split('**API Request Mapping:**')
    cot_context = parts[0].strip()  # We keep the CoT internally, but no longer return it to the UI.
    api_request_mapping = {}

    # Check if there is anything after 'API Request Mapping:'
    if len(parts) < 2:
        print("No 'API Request Mapping' section found in model output.")
        return cot_context, api_request_mapping

    text_after_mapping_label = parts[1]

    # 1) Attempt: explicit code block in format `json ... `
    match = re.search(r"`json\s*(\{.*?\})\s*`", text_after_mapping_label, re.DOTALL)
    if match:
        raw_json_text = match.group(1).strip()
        try:
            api_request_mapping = json.loads(raw_json_text)
            return cot_context, api_request_mapping
        except json.JSONDecodeError:
            print("Error decoding JSON from code block. Fallback to safe_json_parse.")

    # 2) Fallback: safe_json_parse (searches for the first valid { ... } block)
    parsed_json = safe_json_parse(text_after_mapping_label)
    if parsed_json:
        api_request_mapping = parsed_json
    else:
        print("No valid JSON found in API Request Mapping.")

    return cot_context, api_request_mapping

def canonicalize_request(url: str) -> str:
    # Replace /artists/<something> with /artists/{artist-ID}
    url = re.sub(r'/artists/[^/]+', '/artists/{artist-ID}', url)

    # Replace /albums/<something> with /albums/{album-ID}
    url = re.sub(r'/albums/[^/]+', '/albums/{album-ID}', url)

    return url

# Removes the parameter 'cot_context' (and its usage) from this function
def process_api_requests(api_requests, additional_info, second_model_api_url, model_parameters):
    # (1) Load prompt template:
    prompt_template = load_prompt('select_api_call.txt')

    # (2) JSON with API Requests => e.g., {"Question A": "URL-A", "Question B": "URL-B"}
    api_requests_json_str = json.dumps(api_requests, indent=2)

    # (3) Fill the prompt
    prompt = prompt_template.format(
        api_requests=api_requests_json_str,
        additional_info=additional_info.strip() if additional_info else "None"
    )

    # (4) Call the model
    response = call_model_api(prompt, second_model_api_url, model_parameters)
    print(response)
    return response

def find_question_for_api_request(executed_api_request, api_request_mapping):
    # First, canonicalize the executed request back to placeholder form
    canonical = canonicalize_request(executed_api_request)

    # Now look for a direct match with the canonical form
    for question, api_request in api_request_mapping.items():
        if api_request == canonical:
            return question

    return "No matching question found for the executed API request."

def interpret_api_response(api_request, api_response, question, third_model_api_url, model_parameters):
    prompt_template = load_prompt('interpret_prompt.txt')
    prompt = prompt_template.format(
        api_request=api_request,
        api_response=json.dumps(api_response, indent=2),
        question=question
    )

    # Here the prompt is written to a txt file
    with open("interpret_prompt_output.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    response = call_model_api(prompt, third_model_api_url, model_parameters)
    print(response)
    return response

# We keep this function, but it is no longer used by button click.
def extract_api_request(response):
    # Use regex to find the first occurrence of an API URL in the response
    match = re.search(r'https?://[^\s\'"<>`]+', response)
    if match:
        api_request = match.group(0)
        # Clean up any trailing punctuation or special characters
        api_request = re.sub(r'[.,\'"<>`]+$', '', api_request)
        return api_request.strip()
    else:
        return "No valid API request found."

def execute_api_request(api_url):
    if not api_url.startswith('http'):
        return {"error": "Invalid API URL"}
    try:
        if "api.spotify.com" in api_url:
            token = get_spotify_access_token()
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.get(api_url, headers=headers)
        else:
            response = requests.get(api_url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def generate_model_input(question):
    base_prompt = load_prompt('no_cot_planing_prompt.txt') # Changed prompt file to 'no_cot_planing_prompt.txt' as per NoCoT version
    # Load only text documentation
    api_documentation = load_api_documentation()

    prompt = base_prompt.replace("{user_query}", question).replace("{api_documentation}", api_documentation)
    return prompt

def call_model_api(model_input, model_api_url, model_parameters):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "inputs": model_input,
        "parameters": model_parameters
    }
    response = requests.post(model_api_url, json=payload, headers=headers)
    if response.status_code != 200:
        return f"Error in model API response: {response.status_code} {response.text}"
    return response.json().get('generated_text', 'Error in model API response').strip()

def on_update_click(interpretation, additional_info, extracted_api_request, api_request_mapping):
    # Append the interpretation to the additional information
    if additional_info:
        updated_additional_info = additional_info.strip() + "\n" + interpretation.strip()
    else:
        updated_additional_info = interpretation.strip()

    # Remove the executed API request and question from the mapping
    question_to_remove = find_question_for_api_request(extracted_api_request, api_request_mapping)
    if question_to_remove in api_request_mapping:
        del api_request_mapping[question_to_remove]

    # Update the api_requests_display
    updated_api_requests_display = json.dumps(api_request_mapping, indent=2)

    # Return the updated values
    return updated_additional_info, updated_api_requests_display, api_request_mapping

def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Chain of Thoughts API Request Handler with Interpretation") # Kept the title as it is for consistency, even though it is NoCoT version. Consider renaming it for NoCoT script.

        # Section for User Question (Remove API Doc Type Selection)
        with gr.Row():
            question_input = gr.Textbox(
                label="Enter your question",
                placeholder="What albums has the artist Tame Impala released?",
                lines=2
            )

        # Model API URLs
        with gr.Row():
            model_api_url_input = gr.Textbox(label="Model API URL (First Model)", value="http://localhost:5001/generate", lines=1)
            second_model_api_url_input = gr.Textbox(label="Model API URL (Second Model)", value="http://localhost:5001/generate", lines=1)
            third_model_api_url_input = gr.Textbox(label="Model API URL (Third Model)", value="http://localhost:5001/generate", lines=1)

        # Parameters for First Model (adjusted default values)
        gr.Markdown("### Parameters for First Model")
        with gr.Row():
            max_new_tokens_1 = gr.Slider(label="Max New Tokens", minimum=10, maximum=500, value=450, step=10) # Adjusted default value
            temperature_1 = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            top_p_1 = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95, step=0.05)
            do_sample_1 = gr.Checkbox(label="Do Sample", value=False)

        generate_button = gr.Button("Generate Plan") # Renamed button label from "Generate Chain of Thoughts" to "Generate Plan" for NoCoT

        # -- Textbox for 'Chain of Thoughts Context' REMOVED HERE --
        # cot_context_display = gr.Textbox(label="Chain of Thoughts Context", lines=10, interactive=True)

        api_requests_display = gr.Textbox(label="Extracted API Requests", lines=10, interactive=True)

        # Parameters for Second Model (adjusted default values)
        gr.Markdown("### Parameters for Second Model")
        with gr.Row():
            max_new_tokens_2 = gr.Slider(label="Max New Tokens", minimum=10, maximum=500, value=150, step=10) # Adjusted default value
            temperature_2 = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            top_p_2 = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95, step=0.05)
            do_sample_2 = gr.Checkbox(label="Do Sample", value=False)

        process_button = gr.Button("Process API Requests")
        processed_requests_display = gr.Textbox(label="Processed API Request", lines=5, interactive=True)

        # Removing the "Extract API Request" button
        extracted_request_display = gr.Textbox(
            label="Extracted API Request", lines=2, interactive=True,
            visible=True
        )

        # Execute API Request
        execute_button = gr.Button("Execute API Request")
        api_response_display = gr.JSON(label="API Response")
        api_response_token_count_display = gr.Textbox(label="API Response Token Count", interactive=True)

        # Additional Information
        additional_info_input = gr.Textbox(
            label="Additional Information",
            placeholder="Provide any additional context or extracted information here.",
            lines=3
        )

        # Parameters for Third Model (adjusted default values)
        gr.Markdown("### Parameters for Third Model")
        with gr.Row():
            max_new_tokens_3 = gr.Slider(label="Max New Tokens", minimum=10, maximum=500, value=200, step=10) # Adjusted default value
            temperature_3 = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            top_p_3 = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95, step=0.05)
            do_sample_3 = gr.Checkbox(label="Do Sample", value=False)

        interpret_button = gr.Button("Interpret API Response")
        interpretation_display = gr.Textbox(label="Interpretation (Third Model)", lines=5, interactive=True)

        update_button = gr.Button("Update and Proceed to Next API Request")

        # Define state variables to store api_request_mapping
        api_request_mapping_state = gr.State()
        original_api_request_mapping_state = gr.State()

        # Button Functions
        def on_generate_click(question, model_api_url, max_new_tokens, temperature, top_p, do_sample):
            model_parameters = {
                "max_new_tokens": int(max_new_tokens),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }
            # We receive 'cot_context', but do NOT return it to UI anymore.
            cot_context, api_request_mapping = generate_cot_and_extract_requests(question, model_api_url, model_parameters)
            api_requests_display_value = json.dumps(api_request_mapping, indent=2)
            # Instead of 4 return values (including cot_context_display) now only 3:
            #  - api_requests_display
            #  - api_request_mapping_state
            #  - original_api_request_mapping_state
            return api_requests_display_value, copy.deepcopy(api_request_mapping), api_request_mapping

        # on_process_click no longer needs 'cot_context'
        def on_process_click(api_requests, additional_info, second_model_api_url,
                             max_new_tokens, temperature, top_p, do_sample):
            # 1) parse input (existing JSON from "api_requests_display")
            api_requests_dict = json.loads(api_requests)

            # 2) Model parameters
            model_parameters = {
                "max_new_tokens": int(max_new_tokens),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }

            # 3) Call the model (2nd endpoint) => raw output
            raw_output = process_api_requests(
                api_requests_dict,  # No cot_context anymore
                additional_info,
                second_model_api_url,
                model_parameters
            )

            # 4) First try code block
            match = re.search(r"`json\s*(\{.*?\})\s*`", raw_output, re.DOTALL)
            data = None
            if match:
                raw_json_text = match.group(1).strip()
                try:
                    data = json.loads(raw_json_text)
                except json.JSONDecodeError:
                    print("Error decoding JSON from code block. Fallback to safe_json_parse.")

            # 5) Fallback
            if not data:
                data = safe_json_parse(raw_output)

            # 6) If still no JSON
            if not data:
                return "{}", "No valid request found"

            # 7) data -> 'Processed API Request'
            processed_str = json.dumps(data, indent=2)
            url_list = list(data.values())
            extracted_str = "\n".join(url_list) if url_list else "No URL found"

            return processed_str, extracted_str

        # Since the "Extract API Request" button was removed, this function is no longer used by click.
        def on_execute_click(api_request):
            api_response = execute_api_request(api_request)
            # Convert the API response to a JSON-formatted string
            api_response_str = json.dumps(api_response, indent=2)

            # 1) Count "word tokens" (simple)
            simple_token_count = count_tokens(api_response_str)

            # 2) Count with Gemma-Tokenizer
            encoded_gemma = tokenizer_gemma.encode(api_response_str)
            gemma_count = len(encoded_gemma)

            # 3) Count with Qwen-Tokenizer
            encoded_qwen = tokenizer_qwen.encode(api_response_str)
            qwen_count = len(encoded_qwen)

            # Summarize in a string
            combined_info = (
                f"Word Tokens (simple): {simple_token_count} | "
                f"Gemma-2-27B-it Tokens: {gemma_count} | "
                f"Qwen2.5-7B Tokens: {qwen_count}"
            )

            return api_response, combined_info

        def on_interpret_click(api_request, api_response, original_api_request_mapping, third_model_api_url,
                               max_new_tokens, temperature, top_p, do_sample):
            model_parameters = {
                "max_new_tokens": int(max_new_tokens),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }
            question = find_question_for_api_request(api_request, original_api_request_mapping)
            interpretation = interpret_api_response(api_request, api_response, question, third_model_api_url, model_parameters)
            return interpretation

        def on_update_click_fn(interpretation, additional_info, extracted_api_request, api_request_mapping):
            # We use the existing helper function 'on_update_click' (to avoid confusion, I rename it here)
            updated_additional_info, updated_api_requests_display, updated_mapping = on_update_click(
                interpretation, additional_info, extracted_api_request, api_request_mapping
            )
            return updated_additional_info, updated_api_requests_display, updated_mapping

        # Button links
        generate_button.click(
            on_generate_click,
            inputs=[question_input, model_api_url_input, max_new_tokens_1, temperature_1, top_p_1, do_sample_1],
            # NO 'cot_context_display' anymore in outputs:
            outputs=[api_requests_display, api_request_mapping_state, original_api_request_mapping_state]
        )

        process_button.click(
            on_process_click,
            # NO 'cot_context_display' anymore in inputs
            inputs=[api_requests_display, additional_info_input,
                    second_model_api_url_input, max_new_tokens_2, temperature_2,
                    top_p_2, do_sample_2],
            outputs=[processed_requests_display, extracted_request_display]
        )

        execute_button.click(
            on_execute_click,
            inputs=extracted_request_display,
            outputs=[api_response_display, api_response_token_count_display]
        )

        interpret_button.click(
            on_interpret_click,
            inputs=[extracted_request_display, api_response_display, original_api_request_mapping_state,
                    third_model_api_url_input, max_new_tokens_3, temperature_3, top_p_3, do_sample_3],
            outputs=[interpretation_display]
        )

        update_button.click(
            on_update_click_fn,
            inputs=[interpretation_display, additional_info_input, extracted_request_display, api_request_mapping_state],
            outputs=[additional_info_input, api_requests_display, api_request_mapping_state]
        )

    demo.launch()

if __name__ == "__main__":
    gradio_app()