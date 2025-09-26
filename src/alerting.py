import os
import json
import requests


def post_to_slack(webhook_url: str, text: str) -> None:
    try:
        requests.post(webhook_url, json={"text": text}, timeout=10)
    except Exception:
        pass


def main():
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        return

    decision_path = "metrics/decision.json"
    if not os.path.exists(decision_path):
        return

    with open(decision_path) as f:
        decision = json.load(f)

    meets = decision.get("meets_threshold", False)
    pass_rate = decision.get("pass_rate")
    source = decision.get("source", "metrics.json")

    if not meets:
        text = f":warning: Model failed thresholds. pass_rate={pass_rate:.2f}, source={source}. Triggered tuning."
    else:
        text = f":white_check_mark: Model meets thresholds. pass_rate={pass_rate:.2f}, proceeding to push."

    post_to_slack(webhook, text)


if __name__ == "__main__":
    main()



