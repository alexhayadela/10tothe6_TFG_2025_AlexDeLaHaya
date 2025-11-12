import os
import smtplib
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import  datetime 
from news_rss import relevant_feeds, feeds_to_html


def send_email(text, html):

    # Load .env only if it exists (local run)
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
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
    subject = f"TFG Inversión: Daily status report - {today_date}"
    message['Subject'] = subject

    
    # Add image
    img_path = os.path.join(os.path.dirname(__file__), 'imgs/freakbob.png') 
    with open(img_path, "rb") as f:
        img_data = f.read()
    image = MIMEImage(img_data)
    image.add_header("Content-ID", "<freakbob>")
    message.attach(image)

    start_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>10^6, Noticias y consejos diarios para invertir</title>
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
            <p>Alex De La Haya © {datetime.date.today().year} | 10^6, Noticias y consejos diarios para invertir</p>
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

    df_feeds = relevant_feeds()
    feeds_html = feeds_to_html(df_feeds)
    
    text = "Hello, this is a test mesage"
    send_email(text, feeds_html)



