from flask import Flask, request, jsonify
from flask_cors import CORS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
import base64

app = Flask(__name__)
CORS(app)

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "vishxx3@gmail.com"  # Replace with your email
APP_PASSWORD = "***"      # Replace with your Gmail App Password

@app.route('/send_portfolio_email', methods=['POST'])
def send_portfolio_email():
    try:
        data = request.json
        
        client_name = data.get('client_name')
        client_email = data.get('client_email')
        client_id = data.get('client_id')
        pdf_html = data.get('pdf_html')
        
        if not all([client_name, client_email, client_id, pdf_html]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400
        
        # Create email message
        print (client_email)
        msg = MIMEMultipart('alternative')
        msg['From'] = SENDER_EMAIL
        msg['To'] = client_email
        msg['Subject'] = f'Your Portfolio Report - {datetime.now().strftime("%B %d, %Y")}'
        
        # Email body
        email_body = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                .email-container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .content {{
                    background: #f8fafc;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .button {{
                    display: inline-block;
                    background: #3b82f6;
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 5px;
                    margin-top: 15px;
                }}
                .footer {{
                    text-align: center;
                    color: #64748b;
                    font-size: 12px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <h1>Portfolio Report</h1>
                    <p>Your personalized investment portfolio analysis</p>
                </div>
                
                <div class="content">
                    <h2>Dear {client_name},</h2>
                    <p>We are pleased to share your comprehensive portfolio report. This report includes:</p>
                    <ul>
                        <li>Current portfolio analysis</li>
                        <li>Asset allocation breakdown</li>
                        <li>12-month financial projections</li>
                        <li>AI-powered investment recommendations</li>
                    </ul>
                    <p>Please find your detailed portfolio report attached to this email.</p>
                    <p>If you have any questions or would like to discuss your portfolio, please don't hesitate to reach out to us.</p>
                </div>
                
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                    <p>© {datetime.now().year} Portfolio Management System. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Attach HTML body
        msg.attach(MIMEText(email_body, 'html'))
        
        # Create PDF from HTML (storing HTML as text file for now)
        # In production, you'd use a library like pdfkit or weasyprint
        pdf_filename = f"Portfolio_Report_{client_id}_{datetime.now().strftime('%Y%m%d')}.html"
        
        # Attach the HTML as a file
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(pdf_html.encode('utf-8'))
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename={pdf_filename}')
        msg.attach(attachment)
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        
        return jsonify({
            'status': 'success',
            'message': f'Portfolio report sent successfully to {client_email}'
        }), 200
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({
            'status': 'error',
            'message': 'Email authentication failed. Please check your credentials.'
        }), 500
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to send email: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=8006)