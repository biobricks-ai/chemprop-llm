import dotenv, slack_sdk, time, os

dotenv.load_dotenv()

def send_slack_notification(msg='devnotification'):
    client = slack_sdk.WebClient(token=os.getenv("SLACK_TOKEN"))
    try:
        response = client.chat_postMessage(channel='#tomnotify', text=msg)
        assert response["ok"]
    except Exception as e:
        print(f"Error sending message: {e}")

# Example Usage
def long_running_task():
    time.sleep(10)  # Replace with your actual long-running task
    send_slack_notification()

long_running_task()