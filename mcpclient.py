# client.py
from fastmcp import Client
import asyncio

async def main():
    # Connect to the server
    async with Client("http://localhost:8000/mcp") as client:
        print("Connected to MCP server!")

        # List all available tools
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        print("\n")
        print("CSVdata")

        csvdata= await client.call_tool("get_client_metadata", {"ClientId":"CL001"})
        print(csvdata.data)
        

# Run the client
if __name__ == "__main__":
    asyncio.run(main())