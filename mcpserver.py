# server.py
from fastmcp import FastMCP
import csv

mcp = FastMCP("My Awesome Server")

@mcp.tool
def get_client_metadata(ClientId: str) -> list:
    """Return all data from clientmetadata.csv as a list of dicts"""
    data = []
    with open("./client.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # skip empty rows 
            if row["client_id"].strip() == ClientId.strip():
                data.append(row)          
    return data

if __name__ == "__main__":
    # HTTP transport on port 8000
    mcp.run(transport="http", host="0.0.0.0", port=8000)