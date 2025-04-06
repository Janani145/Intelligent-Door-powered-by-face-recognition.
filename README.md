# Intelligent-Door-powered-by-face-recognition.
This project is a smart security solution designed to automate door unlocking using face recognition and send instant alerts when someone approaches the door. The system uses a webcam to continuously monitor the entrance. When it detects a person, it captures the face from the video stream using Haar Cascade, which is a widely used technique for face detection.

Once the face is detected, it is compared with a pre-saved reference image using SIFT (Scale-Invariant Feature Transform), a feature-matching algorithm that identifies key points in the image. If the face matches the reference image (meaning the person is authorized), the system automatically unlocks the door and sends a welcome message via SMS and email.

If the detected face does not match the authorized image, the system treats the person as unknown. In this case, it immediately sends an alert message along with the captured image to the owner's phone number and email address. This helps in taking quick action if there is an intruder or unauthorized visitor.

The system uses Twilio to send SMS alerts and the SMTP protocol to send emails. It combines computer vision, machine learning, and communication technologies to provide an intelligent and secure access control system.

This project is especially useful when the owner is not at home or is traveling to another city or country. Even if the person is far away, they will still receive instant alerts on their phone and email if someone approaches the door. This helps in monitoring the safety of the home from anywhere in the world and improves peace of mind.

This smart door unlocking and alert system can be used in homes, offices, hostels, and other secure areas to reduce the risk of unauthorized access and improve overall safety.


