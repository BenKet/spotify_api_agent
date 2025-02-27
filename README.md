# Example Project: AI Agents with and without Chain of Thought (CoT)

This repository contains two distinct implementations of an AI agent designed to interact with APIs, specifically focusing on the application of Chain of Thought (CoT) in the planning phase.

## Repository Structure

This repository is organized into two main folders:

*   **`spotify_api_text_models_CoT/`**: This folder contains the AI agent implementation that utilizes Chain of Thought (CoT) during the initial planning process to determine the sequence of API calls needed to answer a user query.
*   **`spotify_api_text_models_NoCoT/`**: (This folder would contain the AI agent implementation that does not use CoT. Based on the provided code, it seems you are currently working on the CoT version. If you create the 'without CoT' version, this README will apply to both.) This folder would house an alternative implementation of the AI agent that attempts to solve user queries without explicitly using Chain of Thought in its planning phase.

The primary objective of this project was to investigate the impact of Chain of Thought in the planning stage of an API-interacting AI agent.  Our core investigation focused on comparing the performance and effectiveness of these two approaches.

## Getting Started

To run the AI agent, follow these steps:

1.  **Clone this repository** to your local machine.

2.  **Navigate to either the `spotify_api_text_models_CoT/` or `spotify_api_text_models_NoCoT/` folder** depending on which version you want to run.

3.  **Environment Configuration:**
    *   Create a `.env` file in the root directory of the chosen agent folder (`spotify_api_text_models_CoT/` or `spotify_api_text_models_NoCoT/`).
    *   Add the following environment variables to your `.env` file. **Important:** You need to obtain these credentials if you plan to use Spotify API functionality, otherwise, the core logic will still function but Spotify-specific requests might fail.

        ```
        CLIENT_ID=YOUR_SPOTIFY_CLIENT_ID
        CLIENT_SECRET=YOUR_SPOTIFY_CLIENT_SECRET
        ```
        Replace `YOUR_SPOTIFY_CLIENT_ID` and `YOUR_SPOTIFY_CLIENT_SECRET` with your actual Spotify API credentials if required.

4.  **Conda Environment Setup (Recommended):**
    *   It is highly recommended to use Conda to manage the Python environment for this project. If you don't have Conda installed, please install it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
    *   **Create a Conda environment** from the provided `requirements.txt` file.  Navigate to the chosen agent folder (e.g., `spotify_api_text_models_CoT/`) in your terminal and run:
        ```bash
        conda create --name ai_agent_env --file requirements.txt
        conda activate ai_agent_env
        ```
        This will create an isolated Conda environment named `ai_agent_env` and install all necessary Python packages listed in `requirements.txt`.
    *   Alternatively, if you prefer to install packages using pip, ensure you have activated the Conda environment first (`conda activate ai_agent_env`) and then run:
        ```bash
        pip install -r requirements.txt
        ```
    *   The `requirements.txt` file lists all the required Python libraries (e.g., `gradio`, `requests`, `transformers`, `python-dotenv`) for this project.

5.  **Language Model Setup:**
    *   This application requires access to one or more Language Models (LLMs) running on a server or accessible via an API endpoint.
    *   You need to ensure that you have an LLM service running and obtain the API endpoint URL.
    *   The Gradio interface allows you to configure **three separate Model API URLs**:
        *   **First Model API URL (Planning - CoT Agent):**  Used for the initial planning and Chain of Thought generation in the `spotify_api_text_models_CoT/` version.
        *   **Second Model API URL (API Request Selection):**  Used for selecting the most relevant API request from a list.
        *   **Third Model API URL (Response Interpretation):** Used to interpret the response received from the executed API call and provide a human-readable answer.
    *   You can use the same model endpoint for all three tasks, or different models for each, depending on your requirements and available resources.
    *   Enter the respective Model API URLs into the designated "Model API URL" textboxes in the Gradio interface. Default URLs are set to `http://localhost:5001/generate`, assuming a local LLM service is running at this address. You will likely need to adjust these to match your LLM setup.

6.  **Run the Gradio Application:**
    *   Ensure your Conda environment `ai_agent_env` is activated (`conda activate ai_agent_env`).
    *   Run the Python script `spotify_ai_agent.py` from your terminal within the chosen agent folder.  For example: `python spotify_ai_agent.py`
    *   Gradio will provide a local URL in the terminal, typically starting with `http://127.0.0.1:` or `http://localhost:`. Open this URL in your web browser to access the AI Agent interface.

## Using the Application

Once the Gradio interface is running in your browser, you can interact with the AI agent:

1.  **Enter your question** in the "Enter your question" textbox. For example: "What albums has the artist Tame Impala released?".
2.  **Configure Model API URLs** if your LLM endpoints are different from the default `http://localhost:5001/generate`.
3.  **Adjust Model Parameters** for each of the three models (First, Second, and Third Model) as needed using the provided sliders and checkboxes. These parameters control the generation behavior of the Language Models.
4.  Click the **"Generate Chain of Thoughts"** button (for the CoT version) to initiate the planning phase. The "Chain of Thoughts Context" and "Extracted API Requests" textboxes will be populated.
5.  Click the **"Process API Requests"** button to have the agent select the most relevant API request(s). The "Processed API Request" and "Extracted API Request" textboxes will be updated.
6.  Click the **"Execute API Request"** button to send the extracted API request to the API. The "API Response" will be displayed in JSON format.
7.  Click the **"Interpret API Response"** button to have the agent interpret the API response into a human-readable answer. The "Interpretation (Third Model)" textbox will show the result.
8.  Use the **"Update and Proceed to Next API Request"** button to incorporate the interpretation into the "Additional Information" and move to processing any remaining API requests (if applicable).

## Results and Observations

Our experiments indicated that the AI agent utilizing Chain of Thought (CoT) in its planning process (`spotify_api_text_models_CoT/`) demonstrated **significantly improved performance** in accurately answering user queries compared to the agent without CoT (`spotify_api_text_models_NoCoT/`). The CoT approach enabled the agent to generate more coherent and effective plans, leading to more successful API interactions and more accurate and relevant answers to user questions.  Further details and specific performance metrics will be documented [in a separate document/section - *optional, if you have detailed results*].

---

This `README.md` provides a starting point for understanding and running the AI agent example project. Please refer to the code and comments within the Python scripts for more detailed information about the implementation.