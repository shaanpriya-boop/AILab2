from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import csv
import os
import shutil
import uvicorn
# client.py


from marketTrend import reason
from langgraphapp import gapp

app = FastAPI()


# ======================================
# CORS CONFIG
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================
# PUBLIC FOLDER SETUP - POINT TO FRONTEND PUBLIC
# ======================================
# Adjust this path to match your frontend's public folder location
FRONTEND_PUBLIC_FOLDER = "../myapp/public"  # Adjust relative path as needed
# Or use absolute path like: "C:/Users/GenAIBLRANCUSR32/I_am_the_advisor/frontend/myapp/public"

os.makedirs(FRONTEND_PUBLIC_FOLDER, exist_ok=True)

# Optional: Still serve files from backend for download
app.mount("/files", StaticFiles(directory=FRONTEND_PUBLIC_FOLDER), name="files")


# ======================================
# UTILITY — EXPORT PORTFOLIO CSV
# ======================================
def export_plan_to_csv(plan, client_id: str):
    output_file = os.path.join(FRONTEND_PUBLIC_FOLDER, f"{client_id}.csv")

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["asset_type", "action", "percentage", "rationale"])

        for txn in plan.transactions:
            writer.writerow([
                txn.asset_type,
                txn.action,
                txn.percentage,
                txn.rationale
            ])

    print("✓ Portfolio CSV created:", output_file)
    return output_file


# ======================================
# MODEL SCHEMAS
# ======================================
class USER_TOPIC(BaseModel):
    topic: str


class ClientRequest(BaseModel):
    client_id: str


# ======================================
# MARKET ANALYSIS ENDPOINT
# ======================================
@app.post("/analyze_market")
def analyze_market(req: USER_TOPIC):
    try:
        print("\n=== MARKET ANALYSIS STARTED ===")
        print("Topic:", req.topic)

        # Run LangChain agent
        result = reason(req.topic)

        # strategy.csv MUST exist in backend root
        if not os.path.exists("strategy.csv"):
            return {
                "status": "error",
                "message": "❌ strategy.csv NOT FOUND in backend folder."
            }

        # Copy strategy.csv → FRONTEND public/
        shutil.copy("strategy.csv", os.path.join(FRONTEND_PUBLIC_FOLDER, "strategy.csv"))
        print("✓ Copied strategy.csv → frontend/public/")

        # Copy strategy_updated.csv → FRONTEND public/ (if generated)
        if os.path.exists("strategy_updated.csv"):
            shutil.copy(
                "strategy_updated.csv",
                os.path.join(FRONTEND_PUBLIC_FOLDER, "strategy_updated.csv")
            )
            print("✓ Copied strategy_updated.csv → frontend/public/")
        else:
            print("⚠ strategy_updated.csv NOT FOUND. Possibly LLM failed.")

        # Create explanations file in FRONTEND public
        explanation_path = os.path.join(FRONTEND_PUBLIC_FOLDER, "explanations.txt")
        with open(explanation_path, "w", encoding="utf-8") as f:
            f.write("This file contains reasoning behind strategy updates.\n")
            f.write("Generated based on AI market analysis.\n")

        print("✓ explanations.txt created in frontend/public/")

        return {
            "status": "success",
            "message": "Market analysis complete. Files saved to frontend public folder.",
            "files_generated": [
                "strategy.csv",
                "strategy_updated.csv",
                "explanations.txt"
            ]
        }

    except Exception as e:
        print("❌ ERROR in /analyze_market:", e)
        return {"status": "error", "message": str(e)}


# ======================================
# PORTFOLIO FOR INDIVIDUAL CLIENT
# ======================================
@app.post("/analyze_client_portfolio")
def analyze_client(req: ClientRequest):
    try:
        client_id = req.client_id
        print(f"\n=== PORTFOLIO GENERATION STARTED for {client_id} ===")

        # If file exists → return same CSV
        existing_csv = os.path.join(FRONTEND_PUBLIC_FOLDER, f"{client_id}.csv")
        if os.path.exists(existing_csv):
            return {
                "status": "success",
                "already_exists": True,
                "csv_file": f"{client_id}.csv",
                "message": "Portfolio already exists."
            }

        # Provide some market text
        marketConditions = (
            "Commodities mixed. Gold stable. Recession moderate. "
            "Equities volatile. Institutional sentiment cautious."
        )

        # Call LangGraph agent
        print("→ Running LangGraph agent...")
        state = gapp.invoke({
            "client_id": client_id,
            "marketCondtions": marketConditions
        })

        plan = state.get("portfolio_plan")

        if not plan:
            raise ValueError("Portfolio plan NOT generated.")

        # Export into FRONTEND public/
        csv_file_path = export_plan_to_csv(plan, client_id)

        return {
            "status": "success",
            "already_exists": False,
            "csv_file": f"{client_id}.csv",
            "message": "Portfolio created in frontend public folder."
        }

    except Exception as e:
        print("❌ ERROR in /analyze_client_portfolio:", e)
        return {"status": "error", "message": str(e)}


# ======================================
# ROOT ENDPOINT
# ======================================
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Backend active.",
        "endpoints": {
            "POST /analyze_market": "Generate strategy files in frontend public",
            "POST /analyze_client_portfolio": "Generate client portfolio in frontend public",
        }
    }


# ======================================
# RUN BACKEND
# ======================================
if __name__ == "__main__":
    print("🚀 API Running at http://localhost:8005")
    print("📁 Frontend Public folder:", os.path.abspath(FRONTEND_PUBLIC_FOLDER))
    uvicorn.run(app, host="0.0.0.0", port=8005)