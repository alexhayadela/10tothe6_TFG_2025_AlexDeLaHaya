import os
import datetime
import smtplib 
from pathlib import Path
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

from utils import load_env
from db.supabase.queries_news import get_recipients
from newsletter.build import build_newsletter

def send_newsletter(text, html, recipients):
    """Sends newsletter with most relevant news from today."""
    sender = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASSWORD")

    # Construct email
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = sender
    message['Bcc'] = ', '.join(recipients)

    today = datetime.date.today()
    subject = f"10**6 Boletín {today.strftime('%d/%m/%Y')}"
    message['Subject'] = subject

    # Add image
    img_path = Path(__file__).resolve().parent.parent / "imgs" / "freakbob.png"
    with open(img_path, "rb") as f:
        img_data = f.read()
    image = MIMEImage(img_data)
    image.add_header("Content-ID", "<freakbob>")
    message.attach(image)

    # Add text
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, message.as_string())

  
if __name__ == "__main__":
    load_env()
    """ Don't process news twice, read from db
    news = last_news()
    classified_news = classify_news(news)
    top_10 = top_news(classified_news,10)
    ready = newsletter_ready(top_10)
    """
    html = build_newsletter()
    print(html)
    text = "Boletín diario 10**6, parte de mi trabajo de final de grado."
    recipients = get_recipients()

    send_newsletter(text, html, recipients)
