import asyncio
from rag_manager import RagManager

async def main():
    file_path = r"D:\farm_vaidya\voice_agent\data\story.txt"
    rag_manager = RagManager(file_path)

    while True:
        user_query = input("\nAsk something (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        answer = await rag_manager.query(user_query)
        print("\n Answer:", answer)

# Check if already inside an event loop
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already running, just create task
            task = main()
            asyncio.ensure_future(task)
        else:
            loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n Exiting gracefully...")
