import os
import smtplib
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import date


def send_email(text):

    # Load .env only if it exists (local run)
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    
    # Load from Github Secrets 
    sender = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASSWORD")
    
    recipients = sender
    # Future newsletter
    """
    recipients = ["destinatario1@gmail.com", "destinatario2@gmail.com"]
    msg['To'] = ', '.join(recipients)
    """

    # Construct email
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = recipients

    today_date = date.today()
    subject = f"TFG Inversión: Daily status report - {today_date}"
    message['Subject'] = subject

    # Add text
    message.attach(MIMEText(text))

    # Add image
    img_path = os.path.join(os.path.dirname(__file__), 'alex.jpg') 
    with open(img_path, "rb") as f:
        img_data = f.read()
    image = MIMEImage(img_data)
    image.add_header("Content-ID", "<alex>", filename = "alex.png")
    message.attach(image)
    
    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, message.as_string())
    
if __name__ == "__main__":

    send_email("Hello, this is a test message")

# FUTURE ADD DASHBOARD IMAGE // complex report
"""
from email.mime.image import MIMEImage

with open("dashboard.png", "rb") as f:
    img_data = f.read()
msg_image = MIMEImage(img_data)
msg_image.add_header("Content-ID", "<dashboard>")
msg.attach(msg_image)
# ADD COMILLAS
html = f
<html>
  <body>
    <h2>Daily KPI Dashboard - {date.today()}</h2>
    <p>See visualization below:</p>
    <img src="cid:dashboard" alt="Dashboard">
  </body>
</html>

msg.attach(MIMEText(html, "html"))
"""


