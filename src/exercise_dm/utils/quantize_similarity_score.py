# This script include a function to quatizate the similarity score

def quatization_label(score):
    if score < 0.125:
        return 0
    elif score >= 0.125 and score < 0.375:
        return 1
    elif score >= 0.375 and score < 0.625:
        return 2
    elif score >= 0.625 and score < 0.875:
        return 3
    else:
        return 4