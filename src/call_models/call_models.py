from call_chatGPT import call_chatGPT
from call_gemini import call_gemini
from call_claude import call_claude


def call_models():
    try:
        call_chatGPT()
    except:
        print("ChatGPT API call failed. Please check your API key and network connection.")
        pass

    try:
        call_gemini()
    except:
        print("Gemini API call failed. Please check your API key and network connection.")
        pass
    try:
        call_claude()
    except:
        print("Claude API call failed. Please check your API key and network connection.")
        pass


if __name__ == "__main__":
    call_models()
