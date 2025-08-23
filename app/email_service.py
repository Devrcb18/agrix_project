import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
from flask import current_app, render_template_string
import logging

class EmailService:
    def __init__(self):
        self.SMTP_SERVER = "smtp.gmail.com"
        self.SMTP_PORT = 587
        self.SENDER_EMAIL = "devrcb1845@gmail.com"
        self.SENDER_PASSWORD = "eexq ckjq nhah eznn"
        self.logger = logging.getLogger(__name__)  # Initialize logger

    def send_email(self, recipient_email, subject, body, html_body=None, attachments=None):
        """
        Send email with both text and HTML content
        """
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.SENDER_EMAIL  # Fixed: Changed from sender_email to SENDER_EMAIL
            message["To"] = recipient_email
            
            # Add text part
            text_part = MIMEText(body, "plain")
            message.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, "html")
                message.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    with open(attachment, "rb") as file:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(file.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(attachment)}'
                        )
                        message.attach(part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:  # Fixed: Changed from smtp_server to SMTP_SERVER
                server.starttls(context=context)
                server.login(self.SENDER_EMAIL, self.SENDER_PASSWORD)  # Fixed: Changed from sender_password to SENDER_PASSWORD
                text = message.as_string()
                server.sendmail(self.SENDER_EMAIL, recipient_email, text)
            
            self.logger.info(f"Email sent successfully to {recipient_email}")
            return True, "Email sent successfully"
            
        except Exception as e:
            self.logger.error(f"Failed to send email to {recipient_email}: {str(e)}")
            return False, f"Failed to send email: {str(e)}"
    
    def send_contact_form_email(self, name, email, subject, message):
        """
        Send contact form submission email
        """
        email_subject = f"AgriX Contact Form: {subject}"
        
        # Text version
        text_body = f"""
New contact form submission from AgriX website:

Name: {name}
Email: {email}
Subject: {subject}
Message:
{message}

Submitted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # HTML version
        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #2E7D32, #4CAF50); padding: 20px; color: white; text-align: center;">
                    <h1>AgriX Contact Form Submission</h1>
                </div>
                <div style="padding: 20px; background: #f8f9fa;">
                    <h2 style="color: #2E7D32;">New Message Received</h2>
                    <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <p><strong>Name:</strong> {name}</p>
                        <p><strong>Email:</strong> <a href="mailto:{email}">{email}</a></p>
                        <p><strong>Subject:</strong> {subject}</p>
                        <div style="margin-top: 20px;">
                            <strong>Message:</strong>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 10px;">
                                {message.replace(chr(10), '<br>')}
                            </div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 20px; color: #666;">
                        <p>Submitted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        # Send to admin
        admin_email = os.environ.get('ADMIN_EMAIL', 'devanshc.shukla@gmail.com')
        return self.send_email(admin_email, email_subject, text_body, html_body)  # Fixed: Changed email_subject to email_subject

# Initialize email service
email_service = EmailService()