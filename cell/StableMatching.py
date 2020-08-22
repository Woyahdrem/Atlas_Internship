import operator


# Algorithm to compute a stable matching with threshold
def stable_matching(preferences, threshold):
    # Initialize the two groups
    groupA, listA, groupB, listB = prepare_lists(preferences, threshold)

    # The algorithm continues for as long as a None element exists in the group
    i = 1
    while None in groupA.values():
        print(f"---Stating round {i}---")
        print(f"Currently:\n\t{len([k for k in groupA.keys() if groupA[k] is not None])} elements are matched")
        print(f"\t{len([k for k in groupA.keys() if groupA[k] is None])} elements from group A are not matched")
        print(f"\t{len([k for k in groupB.keys() if groupB[k] is None])} elements from group B are not matched")
        do_round(groupA, groupB, listA, listB)

    # Remove the elements without a match
    idx = [i for i, j in groupA.items() if j == -1]
    for i in idx:
        del groupA[i]
    print(f"Final matching: {groupA.items()}")

    idxA = list(groupA.keys())
    idxB = list(groupA.values())

    return idxA, idxB


# Single round of the stable matching algorithm
def do_round(groupA, groupB, listA, listB):
    # Find the elements in groupA without a matching
    proponentsA = [k for k in groupA.keys() if groupA[k] is None]

    for a in proponentsA:
        # Get the element from listA[a] with the highest preference
        prefA = listA[a]

        # If A has no more preferences, it won't be matched with anything
        if len(prefA) == 0:
            print(f"\t\tElement{a} from group A has no more potential matches, removing")
            groupA[a] = -1
            continue

        # Get the most favorite element of A
        b = max(prefA.items(), key=operator.itemgetter(1))[0]
        proposal = prefA[b]
        del prefA[b]

        # A proposes to B
        if not listB[b]:  # B accepts if free
            print(f"\t\tFound new proposal between {a} and {b} with proposal of {proposal}", end="\r")
            # Do proposal
            groupA[a] = b
            groupB[b] = a
            listB[b] = proposal
        elif proposal > listB[b]:  # Or if the proposal is a better one
            # Undo previous proposal
            oldA = groupB[b]
            groupA[oldA] = None
            print(f"Found a better proposal. {oldA} and {b} with proposal of {listB[b]} replaced with {a} and {b} with"
                  f" proposal of {proposal}", end="\r")
            # Do new proposal
            groupA[a] = b
            groupB[b] = a
            listB[b] = proposal


# Function to get the preference between two elements
def get_preference(elemA, elemB, preferences):
    if (elemA, elemB) in preferences.keys():
        return preferences[(elemA, elemB)]
    else:
        return 0.


# Function to prepare the lists for the stable matching
def prepare_lists(preferences, threshold):

    # Get the elements of groupA from the X value of the (X, Y) key of the preferences' keys,
    # then remove the duplicates
    temp = [x for x, y in preferences.keys()]
    temp = list(set(temp))
    groupA = dict.fromkeys(temp)
    # Get the elements of groupB from the Y value of the (X, Y) key of the preferences' keys,
    # then remove the duplicates
    temp = [y for x, y in preferences.keys()]
    temp = list(set(temp))
    groupB = dict.fromkeys(temp)

    listA = {}

    for a in groupA.keys():
        sub_list = {}

        for b in groupB.keys():
            pref = get_preference(a, b, preferences)
            if pref >= threshold:
                sub_list[b] = pref

        listA[a] = sub_list

    listB = dict.fromkeys(groupB.keys())

    return groupA, listA, groupB, listB
