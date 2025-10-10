import requests

# Replace with your ngrok URL
ngrok_url = "https://milton-unextenuated-youthfully.ngrok-free.dev/retrieve_summarized-meeting"

# Sample meeting text to test
meeting_text = """The meeting began with a discussion on the upcoming launch of the company’s new mobile application, focusing on its current development status and the adjustments needed before release. The team lead emphasized that although most of the backend systems were complete, the user interface still required improvements based on the feedback gathered from the beta testers. Members agreed that the onboarding process needed to be simplified to reduce user drop-off rates. The marketing department shared their proposed campaign strategy, suggesting the use of short promotional videos and collaborations with tech influencers to build anticipation before the official launch. Concerns were raised about the project timeline since some features, such as the in-app messaging system, were still under review for security vulnerabilities. The IT head assured the group that additional developers would be temporarily assigned to help resolve these issues within the week. The finance officer reminded everyone to stay within the approved budget, especially since new ad placements and influencer partnerships could increase costs. A brief debate followed regarding whether to prioritize speed or quality, with most agreeing that a short delay would be better than releasing a product with bugs. The meeting concluded with each department outlining their next steps and agreeing to reconvene in two weeks for a progress update.
"""

# Create a temporary file-like object
files = {'file': ('meeting.txt', meeting_text)}

# Send POST request with file
response = requests.post(ngrok_url, files=files)

# Check response
if response.status_code == 200:
    print("✅ Server response:")
    print(response.json())
else:
    print("❌ Failed:", response.status_code, response.text)
