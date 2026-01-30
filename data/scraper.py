import asyncio
from twscrape import API, gather
from twscrape.logger import set_log_level
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

async def main():
    proxy = "http://192.168.49.1:9099"

    api = API(proxy=proxy)

    await api.pool.add_account(os.getenv("UNAME"), os.getenv("PASSWORD"), os.getenv("EMAIL"), os.getenv("EPASSWORD"), cookies=os.getenv("COOKIES"))

    await api.pool.login_all()
    
    tweets = []
    async for tweet in api.search("lang:fa", limit=1000):
        tweets.append(tweet)
        print(tweet.id, tweet.rawContent[:50])
    
    import json
    with open("tweets_fa.jsonl", "w", encoding="utf-8") as f:
        for tweet in tweets:
            f.write(json.dumps(tweet.dict(), ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())