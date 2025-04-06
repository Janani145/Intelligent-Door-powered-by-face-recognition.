import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from twilio.rest import Client
import time
print("Welcome Janani project")
# Twilio setup for SMS
account_sid = "Your ID "
auth_token = "Yout token"
twilio_number = "some number"
recipient_number = "number"
client = Client(account_sid, auth_token)

# Email setup
smtp_server = 'smtp.gmail.com'
smtp_port = 587
sender_email = "some@gmail.com"
sender_password = "password"  # Use App Password instead of Gmail password
recipient_email = "some@gmail.com"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load reference image and compute SIFT descriptors
reference_image = cv2.imread("imgg.jpg", cv2.IMREAD_GRAYSCALE)  # Load as grayscale
sift = cv2.SIFT_create()
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)

def match_faces(img1):
    """Compare detected face with the reference image using SIFT feature matching."""
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)

    if descriptors1 is None or descriptors_ref is None:
        print("❌ No descriptors found. Face matching failed.")
        return False

    # Use BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors_ref, k=2)

    # Apply Lowe’s Ratio Test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    # Dynamic threshold: At least 10% of keypoints should match
    threshold = max(0.1 * len(descriptors1), 10)
    print(f"✅ Good Matches: {len(good_matches)} / {threshold}")

    print(len(good_matches) > threshold)
    return len(good_matches)>5

def send_alerts(image_path, message):
    """Send SMS and email alerts."""
    send_sms_alert(message)
    send_email_alert(image_path, message)

def send_sms_alert(message):
    """Send an SMS alert using Twilio."""
    message = client.messages.create(
        body=message,
        from_=twilio_number,
        to=recipient_number
    )
    print(f"📩 SMS sent: {message.sid}")

def send_email_alert(image_path, message):
    """Send an email with the detected face image."""
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Alert: Face Detected"
    msg.attach(MIMEText(message, 'plain'))
    
    with open(image_path, 'rb') as f:
        img_data = f.read()
    image = MIMEImage(img_data, name="detected_face.jpg")
    msg.attach(image)
    
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, recipient_email, msg.as_string())
    server.quit()
    print("📧 Email sent successfully!")

def detect_faces_webcam():
    """Detect faces from the webcam and check for matching."""
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y + h, x:x + w]  # Extract detected face region
            
            # Save detected face for alert
            image_path = "detected_face.jpg"
            cv2.imwrite(image_path, frame)
            
            if match_faces(face_roi_gray):
                print("✅ Face Matched! Unlocking Door...")
                send_alerts(image_path, "Welcome! Door unlocked.")
                # Add door unlocking mechanism here
            else:
                print("❌ Unknown Face Detected! Sending Alert...")
                send_alerts(image_path, "Unknown person detected! Alerting authorities.")
                
            cap.release()
            cv2.destroyAllWindows()
            return
        
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start face detection from webcam
detect_faces_webcam()   
