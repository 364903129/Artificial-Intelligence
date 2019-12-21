"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None


def question2():
    """
    [lower the noise, since up and down are all (-100)]
    """

    answerDiscount = 0.9
    answerNoise = 0.01

    return answerDiscount, answerNoise


def question3a():
    """
    [since risking the cliff, the noise should be very low rate to
    reduce fall off the cliff. Also exit at close exit, so the living
    reward can set higher, discount can be lower]
    """

    answerDiscount = 0.1
    answerNoise = 0.00001
    answerLivingReward = 0.7

    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    """
    [since not risking the cliff, the noise can set higher a little bit.
    also exit at close exit, the living reward can set higher, discount can be
    lower]
    """

    answerDiscount = 0.1
    answerNoise = 0.001
    answerLivingReward = 0.7

    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    """
    [risking the cliff, set the noise lower to reduce the risk, far exit,
    high discount, low living reward]
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 0.2

    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    """
    [not risk the cliff, higher noise, far exit, high discount low living reward]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.2

    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    """
    [avoid everything, high noise and discount, low living reward]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.1

    return answerDiscount, answerNoise, answerLivingReward


def question6():
    """
    [return Not possible, dont need them]
    """
    return NOT_POSSIBLE


if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
