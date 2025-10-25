def get_relation(boxA, boxB):
    xA, yA = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    xB, yB = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
    dx, dy = abs(xA - xB), abs(yA - yB)

    if dy < 40 and dx < 200:
        return "beside"
    elif yA < yB and dx < 150:
        return "above"
    elif yA > yB and dx < 150:
        return "below"
    elif dx < 300 and dy < 200:
        return "near"
    else:
        return None


def build_scene_graph(detections):
    scene_graph = {"objects": [], "relations": []}
    scene_graph["objects"] = list({d["label"] for d in detections})

    for i in range(len(detections)):
        for j in range(len(detections)):
            if i == j:
                continue
            rel = get_relation(detections[i]["bbox"], detections[j]["bbox"])
            if rel:
                scene_graph["relations"].append(
                    (detections[i]["label"], rel, detections[j]["label"])
                )
    return scene_graph