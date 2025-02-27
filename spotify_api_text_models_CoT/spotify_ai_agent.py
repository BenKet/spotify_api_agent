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
    # First, remove all code fences (`...`) to avoid parsing JSON within code blocks
    text_no_fences = re.sub(r"`.*?`", "", text, flags=re.DOTALL)

    # Find all { ... } blocks (greedy or non-greedy) in the text without code fences:
    json_candidates = re.findall(r"\{.*?\}", text_no_fences, flags=re.DOTALL)

    return json_candidates

def safe_json_parse(model_output: str):
    """
    - Removes code fences from the model output.
    - Finds all JSON blocks (e.g., {...}) in the output.
    - Attempts to parse the first valid JSON block.
    - Returns the parsed Python dictionary if successful, otherwise None.
    """
    candidates = extract_json_blocks(model_output)
    if not candidates:
        return None

    # Iterate through each found JSON block candidate
    for candidate in candidates:
        candidate = candidate.strip() # Remove leading/trailing whitespace
        try:
            parsed = json.loads(candidate) # Attempt to parse the candidate as JSON
            return parsed # If parsing is successful, return the parsed dictionary
        except json.JSONDecodeError:
            # If this block is not parsable as JSON, continue to the next candidate
            continue

    # If none of the JSON blocks were successfully parsed, return None
    return None

def load_env_variables():
    """
    Loads environment variables from a .env file located in the parent directory.
    Retrieves CLIENT_ID and CLIENT_SECRET, and raises ValueError if not found.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory of the script's directory
    parent_dir = os.path.dirname(script_dir)
    # Construct the path to the .env file in the parent directory
    dotenv_path = os.path.join(parent_dir, '.env')

    # Load environment variables from the specified .env file path
    if not load_dotenv(dotenv_path=dotenv_path): # Pass the constructed path to load_dotenv
        raise ValueError(".env file could not be loaded. Ensure it exists and is correctly formatted in the parent directory.")
    client_id = os.getenv('CLIENT_ID') # Retrieve CLIENT_ID from environment variables
    client_secret = os.getenv('CLIENT_SECRET') # Retrieve CLIENT_SECRET from environment variables
    if not client_id or client_secret is None:
        raise ValueError("CLIENT_ID or CLIENT_SECRET not set.") # Raise error if CLIENT_ID or CLIENT_SECRET is not set
    return client_id, client_secret # Return client ID and secret if loading is successful

def get_spotify_access_token():
    """
    Fetches a Spotify API access token using client credentials.
    Uses CLIENT_ID and CLIENT_SECRET from environment variables.
    """
    client_id, client_secret = load_env_variables() # Load client ID and secret from env variables

    auth_url = 'https://accounts.spotify.com/api/token' # Spotify API token endpoint
    auth_header = {
        'Authorization': f'Basic {base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()}' # Base64 encode client ID and secret for Authorization header
    }
    auth_data = {
        'grant_type': 'client_credentials' # Set grant type for client credentials flow
    }

    response = requests.post(auth_url, headers=auth_header, data=auth_data) # Send POST request to Spotify API token endpoint

    if response.status_code != 200:
        raise Exception(f"Error fetching access token: {response.status_code} {response.text}") # Raise exception if request fails

    token_info = response.json() # Parse JSON response
    access_token = token_info.get('access_token') # Extract access token from response
    if not access_token:
        raise Exception("No access token received.") # Raise exception if access token is not found in the response

    return access_token # Return the fetched Spotify access token

# Load the prompt from a file
def load_prompt(file_path):
    """
    Loads a prompt from a text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f: # Open the file in read mode with UTF-8 encoding
        return f.read() # Read and return the content of the file

# Load the API documentation
def load_api_documentation():
    """
    Loads API documentation from 'api_doc_text.txt'.
    This version uses only text documentation as per requirement.
    """
    with open('api_doc_text.txt', 'r', encoding='utf-8') as f: # Open 'api_doc_text.txt' in read mode with UTF-8 encoding
        return f.read() # Read and return the content of the API documentation file

# Function to count tokens (simple word tokenizing, for demo purposes)
def count_tokens(text):
    """
    Counts tokens in a text using simple whitespace splitting.
    For demonstration purposes only, not for accurate token counting.
    """
    tokens = text.split() # Split the text into words based on whitespace
    return len(tokens) # Return the number of tokens

def generate_cot_and_extract_requests(question, model_api_url, model_parameters):
    """
    1) Sends a request to the language model with the user 'question' and planning 'prompt'.
    2) Splits the model's output into 'Chain of Thought' (cot_context) and 'API Request Mapping'
       sections, delimited by '**API Request Mapping:**'.
    3) Extracts API requests from the 'API Request Mapping' section by searching for a JSON block.
       Prioritizes JSON blocks enclosed in `json ... ` code fences,
       and falls back to safe_json_parse if no fenced block is found or parsing fails.
    """
    prompt = generate_model_input(question) # Generate the prompt for the language model using user question
    model_output = call_model_api(prompt, model_api_url, model_parameters) # Call the language model API
    print(model_output) # Print the raw model output for debugging

    # Split the model output at '**API Request Mapping:**' to separate CoT context and API request mapping
    parts = model_output.split('**API Request Mapping:**')
    cot_context = parts[0].strip() # The part before 'API Request Mapping:' is considered Chain of Thought context, and whitespace is removed
    api_request_mapping = {} # Initialize an empty dictionary to store extracted API request mappings

    # Check if there's a section after 'API Request Mapping:' in the model output
    if len(parts) < 2:
        print("No 'API Request Mapping' section found in model output.")
        return cot_context, api_request_mapping # If no API Request Mapping section, return empty mapping

    text_after_mapping_label = parts[1] # The part after 'API Request Mapping:' label

    # 1) Attempt to extract JSON from an explicit code block with `json ... ` format
    match = re.search(r"`json\s*(\{.*?\})\s*`", text_after_mapping_label, re.DOTALL)
    if match:
        raw_json_text = match.group(1).strip() # Extract the JSON text from the code block and remove whitespace
        try:
            api_request_mapping = json.loads(raw_json_text) # Attempt to parse the extracted text as JSON
            return cot_context, api_request_mapping # If successful, return CoT context and API request mapping
        except json.JSONDecodeError:
            print("Error decoding JSON from code block. Fallback to safe_json_parse.") # Log error and fallback to general JSON parsing

    # 2) Fallback: Use safe_json_parse to find and parse any JSON block in the 'API Request Mapping' section
    parsed_json = safe_json_parse(text_after_mapping_label)
    if parsed_json:
        api_request_mapping = parsed_json # If safe_json_parse finds valid JSON, use it as API request mapping
    else:
        print("No valid JSON found in API Request Mapping.") # Log if no valid JSON is found

    return cot_context, api_request_mapping # Return CoT context and API request mapping (could be empty if no JSON was parsed)

def canonicalize_request(url: str) -> str:
    """
    Canonicalizes API request URLs by replacing specific IDs with placeholders.
    This is used to match executed requests back to the originally planned requests,
    which may contain placeholders instead of concrete IDs.
    """
    # Replace /artists/<artist-ID> with /artists/{artist-ID} - generic artist ID placeholder
    url = re.sub(r'/artists/[^/]+', '/artists/{artist-ID}', url)

    # Replace /albums/<album-ID> with /albums/{album-ID} - generic album ID placeholder
    url = re.sub(r'/albums/[^/]+', '/albums/{album-ID}', url)

    return url # Return the canonicalized URL with placeholders

def process_api_requests(api_requests, cot_context, additional_info, second_model_api_url, model_parameters):
    """
    Processes extracted API requests using a second language model.
    Selects the most relevant API call based on the API requests and additional information.
    """
    # (1) Load prompt template for selecting the API call
    prompt_template = load_prompt('select_api_call.txt')

    # (2) Convert the API requests dictionary into a JSON string for the prompt
    api_requests_json_str = json.dumps(api_requests, indent=2)

    # (3) Format the prompt with API requests, CoT context, and additional information
    prompt = prompt_template.format(
        api_requests=api_requests_json_str,
        additional_info=additional_info.strip() if additional_info else "None" # Use "None" if no additional info provided
    )

    # (4) Call the second language model API with the formatted prompt
    response = call_model_api(prompt, second_model_api_url, model_parameters)
    print(response) # Print the raw response from the second model for debugging
    return response # Return the response from the second model

def find_question_for_api_request(executed_api_request, api_request_mapping):
    """
    Finds the original question associated with an executed API request.
    This is done by canonicalizing the executed request and matching it against
    the API request mapping generated earlier, which links questions to API requests.
    """
    # First, canonicalize the executed request to replace IDs with placeholders for matching
    canonical = canonicalize_request(executed_api_request)

    # Now iterate through the API request mapping to find a match
    for question, api_request in api_request_mapping.items():
        if api_request == canonical: # Check if the canonicalized executed request matches any API request in the mapping
            return question # If a match is found, return the corresponding question

    return "No matching question found for the executed API request." # If no match is found, return a default message

def interpret_api_response(api_request, api_response, question, third_model_api_url, model_parameters):
    """
    Interprets the API response using a third language model to provide a human-readable answer.
    Uses the original API request, the raw API response, and the user's question to generate the interpretation.
    """
    prompt_template = load_prompt('interpret_prompt.txt') # Load the prompt template for interpretation
    prompt = prompt_template.format(
        api_request=api_request,
        api_response=json.dumps(api_response, indent=2), # Convert API response to JSON string for prompt
        question=question
    )

    # Write the complete interpretation prompt to a file for debugging and logging
    with open("interpret_prompt_output.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    response = call_model_api(prompt, third_model_api_url, model_parameters) # Call the third language model API for interpretation
    print(response) # Print the raw response from the third model for debugging
    return response # Return the interpretation generated by the third model

# We keep this function, but it is no longer used by a button click in the UI.
def extract_api_request(response):
    """
    Extracts an API request URL from a text response using regular expressions.
    This function is kept for potential internal use, but is not directly triggered by a UI button anymore.
    """
    # Use regex to find the first occurrence of an API URL in the response
    match = re.search(r'https?://[^\s\'"<>`]+', response)
    if match:
        api_request = match.group(0) # Extract the matched URL
        # Clean up any trailing punctuation or special characters from the URL
        api_request = re.sub(r'[.,\'"<>`]+$', '', api_request)
        return api_request.strip() # Return the cleaned and stripped API request URL
    else:
        return "No valid API request found." # Return message if no API request URL is found

def execute_api_request(api_url):
    """
    Executes an API request to a given URL.
    Handles Spotify API requests with token authentication and generic requests without.
    Returns the JSON response from the API or an error message.
    """
    if not api_url.startswith('http'):
        return {"error": "Invalid API URL"} # Return error if the URL doesn't start with 'http'
    try:
        if "api.spotify.com" in api_url:
            token = get_spotify_access_token() # Get Spotify access token for Spotify API requests
            headers = {'Authorization': f'Bearer {token}'} # Set authorization header with Bearer token
            response = requests.get(api_url, headers=headers) # Send GET request to Spotify API with authorization header
        else:
            response = requests.get(api_url) # Send GET request to generic API without special headers
        return response.json() # Parse and return the JSON response from the API
    except Exception as e:
        return {"error": str(e)} # Return error message in JSON format if any exception occurs during the API call

def generate_model_input(question):
    """
    Generates the input prompt for the language model based on the user question and API documentation.
    Loads the prompt template and API documentation from files, and substitutes placeholders.
    """
    base_prompt = load_prompt('cot_planing_prompt.txt') # Load the base prompt template from 'cot_planing_prompt.txt' for CoT planning
    # Load only text documentation - as per current requirements
    api_documentation = load_api_documentation() # Load API documentation text

    prompt = base_prompt.replace("{user_query}", question).replace("{api_documentation}", api_documentation) # Replace placeholders in the base prompt with user query and API documentation
    print(prompt) # Print the generated prompt for debugging
    return prompt # Return the final prompt string

def call_model_api(model_input, model_api_url, model_parameters):
    """
    Calls a language model API endpoint with the given model input and parameters.
    Returns the generated text from the model or an error message.
    """
    headers = {'Content-Type': 'application/json'} # Set headers for JSON content type
    payload = {
        "inputs": model_input, # Input text for the language model
        "parameters": model_parameters # Model parameters (e.g., max_new_tokens, temperature)
    }
    response = requests.post(model_api_url, json=payload, headers=headers) # Send POST request to the model API endpoint
    if response.status_code != 200:
        return f"Error in model API response: {response.status_code} {response.text}" # Return error message if API call fails
    return response.json().get('generated_text', 'Error in model API response').strip() # Extract and return the generated text from the JSON response, or an error message if not found

def on_update_click(interpretation, additional_info, extracted_api_request, api_request_mapping):
    """
    Handles the 'Update and Proceed to Next API Request' button click.
    Updates the additional information with the latest interpretation, and removes the executed API request
    from the API request mapping, preparing for the next API request processing if any are remaining.
    """
    # Append the interpretation to the additional information text box
    if additional_info:
        updated_additional_info = additional_info.strip() + "\n" + interpretation.strip() # Append interpretation to existing additional info, with newline
    else:
        updated_additional_info = interpretation.strip() # If no existing info, use just the interpretation

    # Remove the executed API request and its corresponding question from the api_request_mapping
    question_to_remove = find_question_for_api_request(extracted_api_request, api_request_mapping) # Find the question related to the executed API request
    if question_to_remove in api_request_mapping:
        del api_request_mapping[question_to_remove] # Delete the question and its API request from the mapping

    # Update the api_requests_display textbox with the modified api_request_mapping (remaining requests)
    updated_api_requests_display = json.dumps(api_request_mapping, indent=2)

    # Return the updated values for additional information, API requests display, and the API request mapping state
    return updated_additional_info, updated_api_requests_display, api_request_mapping

def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Chain of Thoughts API Request Handler with Interpretation")

        # Section for User Question (API Doc Type Auswahl entfernen)
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

        # Parameters for First Model (angepasste Defaultwerte)
        gr.Markdown("### Parameters for First Model")
        with gr.Row():
            max_new_tokens_1 = gr.Slider(label="Max New Tokens", minimum=10, maximum=800, value=600, step=10)
            temperature_1 = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            top_p_1 = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95, step=0.05)
            do_sample_1 = gr.Checkbox(label="Do Sample", value=False)
        generate_button = gr.Button("Generate Chain of Thoughts")
        cot_context_display = gr.Textbox(label="Chain of Thoughts Context", lines=10, interactive=True)
        api_requests_display = gr.Textbox(label="Extracted API Requests", lines=10, interactive=True)

        # Parameters for Second Model (angepasste Defaultwerte)
        gr.Markdown("### Parameters for Second Model")
        with gr.Row():
            max_new_tokens_2 = gr.Slider(label="Max New Tokens", minimum=10, maximum=500, value=250, step=10)
            temperature_2 = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            top_p_2 = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95, step=0.05)
            do_sample_2 = gr.Checkbox(label="Do Sample", value=False)

        process_button = gr.Button("Process API Requests") # Button Definition **MOVED UP**
        processed_requests_display = gr.Textbox(label="Processed API Request", lines=5, interactive=True) # Textbox Definition **MOVED UP**
        extracted_request_display = gr.Textbox( # Textbox Definition **MOVED UP**
            label="Extracted API Request", lines=2, interactive=True,
            visible=True
        )

        # Execute API Request - Definitions **MOVED UP**
        execute_button = gr.Button("Execute API Request") # Button Definition **MOVED UP**
        api_response_display = gr.JSON(label="API Response") # JSON Display Definition **MOVED UP**
        api_response_token_count_display = gr.Textbox(label="API Response Token Count", interactive=True) # Textbox Definition **MOVED UP**

        # Additional Information
        additional_info_input = gr.Textbox(
            label="Additional Information",
            placeholder="Provide any additional context or extracted information here.",
            lines=3
        )

        # Parameters for Third Model (angepasste Defaultwerte)
        gr.Markdown("### Parameters for Third Model")
        with gr.Row():
            max_new_tokens_3 = gr.Slider(label="Max New Tokens", minimum=10, maximum=500, value=300, step=10)
            temperature_3 = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            top_p_3 = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.95, step=0.05)
            do_sample_3 = gr.Checkbox(label="Do Sample", value=False)
        interpret_button = gr.Button("Interpret API Response")
        interpretation_display = gr.Textbox(label="Interpretation (Third Model)", lines=5, interactive=True)

        # Add the new button
        update_button = gr.Button("Update and Proceed to Next API Request")

        # Define state variables to store api_request_mapping
        api_request_mapping_state = gr.State()
        original_api_request_mapping_state = gr.State()

        # Button Functions (moved before button.click calls)
        def on_generate_click(question, model_api_url, max_new_tokens, temperature, top_p, do_sample):
            model_parameters = {
                "max_new_tokens": int(max_new_tokens),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }
            model_output, api_request_mapping = generate_cot_and_extract_requests(question, model_api_url, model_parameters)
            api_requests_display_value = json.dumps(api_request_mapping, indent=2)
            return model_output, api_requests_display_value, copy.deepcopy(api_request_mapping), api_request_mapping

        def on_process_click(api_requests, cot_context, additional_info, second_model_api_url,
                             max_new_tokens, temperature, top_p, do_sample):
            # 1) parse input (vorhandenes JSON aus "api_requests_display")
            api_requests_dict = json.loads(api_requests)

            # 2) Modell-Parameter
            model_parameters = {
                "max_new_tokens": int(max_new_tokens),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }

            # 3) Rufe das Modell (2. Endpunkt) => Roh-Output
            raw_output = process_api_requests(
                api_requests_dict, cot_context, additional_info,
                second_model_api_url, model_parameters
            )

            # 4) Versuche zuerst Code-Block
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

            # 6) Falls gar kein JSON
            if not data:
                return "{}", "No valid request found"

            # 7) data -> 'Processed API Request'
            processed_str = json.dumps(data, indent=2)
            url_list = list(data.values())
            extracted_str = "\n".join(url_list) if url_list else "No URL found"

            return processed_str, extracted_str

        # Da der "Extract API Request"-Button entfernt wurde, wird diese Funktion nicht mehr per Klick genutzt.
        # Wir behalten sie aber (falls intern benötigt).
        def on_execute_click(api_request):
            api_response = execute_api_request(api_request)
            # Convert the API response to a JSON-formatted string
            api_response_str = json.dumps(api_response, indent=2)

            # 1) Zählen der "Wort-Tokens"
            simple_token_count = count_tokens(api_response_str)

            # 2) Zählen mit Gemma-Tokenizer
            encoded_gemma = tokenizer_gemma.encode(api_response_str)
            gemma_count = len(encoded_gemma)

            # 3) Zählen mit Qwen-Tokenizer
            encoded_qwen = tokenizer_qwen.encode(api_response_str)
            qwen_count = len(encoded_qwen)

            # Zusammenfassen in einem String
            combined_info = (
                f"Wort-Tokens (einfach): {simple_token_count} | "
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

        def on_update_click(interpretation, additional_info, extracted_api_request, api_request_mapping):
            updated_additional_info, updated_api_requests_display, updated_mapping = on_update_click_inner(
                interpretation, additional_info, extracted_api_request, api_request_mapping
            )
            return updated_additional_info, updated_api_requests_display, updated_mapping

        # Wir kapseln die bisherige Logik in eine Hilfsfunktion, um Doppelung zu vermeiden
        def on_update_click_inner(interpretation, additional_info, extracted_api_request, api_request_mapping):
            if additional_info:
                updated_additional_info = additional_info.strip() + "\n" + interpretation.strip()
            else:
                updated_additional_info = interpretation.strip()

            question_to_remove = find_question_for_api_request(extracted_api_request, api_request_mapping)
            if question_to_remove in api_request_mapping:
                del api_request_mapping[question_to_remove]

            updated_api_requests_display = json.dumps(api_request_mapping, indent=2)
            return updated_additional_info, updated_api_requests_display, api_request_mapping


        # Button-Verknüpfungen (moved below function definitions)
        generate_button.click(
            on_generate_click,
            inputs=[question_input, model_api_url_input, max_new_tokens_1, temperature_1, top_p_1, do_sample_1],
            outputs=[cot_context_display, api_requests_display, api_request_mapping_state, original_api_request_mapping_state]
        )

        process_button.click(
            on_process_click,
            inputs=[api_requests_display, cot_context_display, additional_info_input,
                    second_model_api_url_input, max_new_tokens_2, temperature_2,
                    top_p_2, do_sample_2],
            outputs=[processed_requests_display, extracted_request_display]
        )

        # Der "Extract API Request"-Button ist entfernt => kein click-Aufruf hier

        execute_button.click( # execute_button.click call is now AFTER execute_button is defined
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
            on_update_click,
            inputs=[interpretation_display, additional_info_input, extracted_request_display, api_request_mapping_state],
            outputs=[additional_info_input, api_requests_display, api_request_mapping_state]
        )

    demo.launch()

if __name__ == "__main__":
    gradio_app() # Run the Gradio app if the script is executed directly