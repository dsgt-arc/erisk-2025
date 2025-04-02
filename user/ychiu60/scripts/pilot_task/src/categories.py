"""
Depression indicator categories based on the Beck Depression Inventory-II (BDI-II).
Adapted for chat-based assessment with simplified scoring.
"""

class BDICategories:
    # Multiplier to convert raw chat indicators to BDI-II equivalent score
    CHAT_SEVERITY_MULTIPLIER = 3
    
    # Severity thresholds based on BDI-II scoring
    SEVERITY_THRESHOLDS = {
        "minimal": (0, 13),
        "mild": (14, 19),
        "moderate": (20, 28),
        "severe": (29, 63)
    }
    
    # Critical indicators that require immediate attention
    CRITICAL_INDICATORS = {
        "suicide": 2,  # Level 2 or higher for suicidal thoughts
        "self_harm": 2  # Level 2 or higher for self-harm thoughts
    }
    
    # Depression indicators and their associated phrases/words
    INDICATORS = {
        "sadness": {
            1: ["feel sad", "feeling down", "unhappy"],
            2: ["always sad", "very sad", "can't shake off sadness"],
            3: ["unbearably sad", "extremely depressed", "completely sad"]
        },
        "pessimism": {
            1: ["discouraged", "not optimistic", "worried about future"],
            2: ["no hope", "nothing to look forward to", "future is hopeless"],
            3: ["completely hopeless", "things will never improve", "future is doomed"]
        },
        "past_failure": {
            1: ["disappointed in myself", "made mistakes", "failed at things"],
            2: ["many failures", "failed more than others", "accomplished very little"],
            3: ["complete failure", "totally failed", "never succeeded at anything"]
        },
        "loss_of_pleasure": {
            1: ["less enjoyment", "don't enjoy things as much", "less fun"],
            2: ["little pleasure", "rarely enjoy anything", "nothing is fun"],
            3: ["no pleasure at all", "can't enjoy anything", "completely lost interest"]
        },
        "guilt": {
            1: ["feel guilty", "blame myself", "regret things"],
            2: ["very guilty", "blame myself often", "deserve punishment"],
            3: ["always guilty", "terrible person", "deserve to suffer"]
        },
        "self_dislike": {
            1: ["disappointed in myself", "don't like myself", "critical of myself"],
            2: ["hate myself", "blame myself", "many faults"],
            3: ["completely worthless", "hate myself entirely", "totally inadequate"]
        },
        "suicide": {
            1: ["life is not worth living", "wish I was dead", "rather be dead"],
            2: ["think about suicide", "want to kill myself", "suicidal thoughts"],
            3: ["will commit suicide", "plan to kill myself", "going to end it"]
        },
        "self_harm": {
            1: ["want to hurt myself", "thoughts of harming myself", "deserve pain"],
            2: ["plan to hurt myself", "going to harm myself", "need to punish myself"],
            3: ["actively self-harming", "cutting myself", "burning myself"]
        }
    }
    
    # Recommendations based on severity
    RECOMMENDATIONS = {
        "minimal": [
            "Continue to monitor your feelings and maintain healthy habits.",
            "Practice self-care and engage in activities you enjoy.",
            "Reach out to friends or family when you need support."
        ],
        "mild": [
            "Consider talking to a counselor or therapist about your feelings.",
            "Establish a regular sleep schedule and exercise routine.",
            "Practice stress-management techniques like meditation or deep breathing."
        ],
        "moderate": [
            "Strongly recommend scheduling an appointment with a mental health professional.",
            "Tell someone you trust about how you're feeling.",
            "Create a daily routine to help structure your day."
        ],
        "severe": [
            "Please seek professional help immediately.",
            "Contact a crisis hotline if you need immediate support.",
            "Don't keep these feelings to yourself - tell someone you trust right away."
        ],
        "critical": [
            "IMPORTANT: Your safety is the top priority. Please seek immediate help.",
            "Call emergency services or go to the nearest emergency room.",
            "Contact the National Suicide Prevention Lifeline at 988 (US)."
        ]
    }
