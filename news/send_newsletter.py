import os
from pathlib import Path
import smtplib
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import datetime 
from news.news_rss import last_news, html_news, newsletter_ready, top_news
from news.classification import classify_news

def send_newsletter(text, html):

    # Load .env only if it exists (local run)
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    
    # Load from Github Secrets 
    sender = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("EMAIL_TARGET_USER")

    # Construct email
    message = MIMEMultipart()
    message['From'] = sender
    recipients = [sender,recipient]
    message['To'] = ', '.join(recipients)

    today_date = datetime.date.today()
    subject = f"10**6 Boletín {today_date}"
    message['Subject'] = subject

    
    # Add image
    img_path = Path(__file__).resolve().parent.parent / "imgs" / "freakbob.png"
    with open(img_path, "rb") as f:
        img_data = f.read()
    image = MIMEImage(img_data)
    image.add_header("Content-ID", "<freakbob>")
    message.attach(image)

    start_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>10**6 Boletín</title>
<style>
  /* Safe email styles (fallback) */
  body {{
      margin:0;
      padding:0;
      background-color:#ffffff;
      font-family: Arial, Helvetica, sans-serif;
  }}
  /* container uses max-width in px and is centered */
  .container {{
      max-width:650px;
      margin:30px auto;           /* centered */
      background-color:#f0f0f0;   /* gentle background for container */
      border-radius:12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      padding: 30px;
      text-align: left;
  }}
  .header {{
      text-align:center;
      margin-bottom:25px;         /* correct property */
  }}
  h1 {{
      text-align:center;
      color:navy;
      font-size: 30px;
      margin-top:0;
      margin-bottom:25px;
  }}
  .news-item {{
      margin-bottom: 20px;
      border-bottom:1px solid #eee;
      padding-bottom: 15px;
  }}
  h2 {{
      color:darkblue;
      font-size: 18px;
      margin-bottom: 5px;
      margin-top:0;
  }}
  p {{
      font-size: 14px;
      line-height:1.6;
      color:#111;
      margin:0 0 8px 0;
  }}
  .footer{{
      text-align:center;
      margin-top:30px;
      font-size:12px;
      color:darkslategray;
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <!-- use cid:... if embedding images inline in email -->
      <img src="cid:freakbob" alt="freakbob" width="479" height="242" style="display:block; margin:0 auto; border-radius:6px;">
    </div>
    <h1>Noticias relevantes de hoy</h1>
"""
    end_html = f"""
    <div class="footer">
            <p>Alex De La Haya © {datetime.date.today().year} | 10**6, Boletín</p>
        </div>
    </div>
</body>
</html>"""
    final_html = start_html + html + end_html

    # Add text
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(final_html, "html"))

    
    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, message.as_string())

# Test    
if __name__ == "__main__":
   
    news = last_news()
    classified_news = classify_news(news)
    top_10 = top_news(classified_news,10)
    ready = newsletter_ready(top_10)
    news_html = html_news(ready)
    
    text = "Boletín diario 10**6, parte de mi trabajo de final de grado."
    send_newsletter(text, news_html)
