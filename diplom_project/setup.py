import cv2

def count_objects_crossing_the_virtual_line(self, cap, line_begin, line_end, targeted_classes=[],
                                            output_path="the_output.avi", show=False):
    ret, frame = cap.read()

    fps, height, width = get_output_fps_height_and_width(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Sort()
    memory = {}

    line = [line_begin, line_end]
    counter = 0

    while ret:

        objects = self.tfnet.return_predict(frame)

        if targeted_classes:
            objects = list(filter(lambda res: res["label"] in targeted_classes, objects))

        results, _ = self._convert_detections_into_list_of_tuples_and_count_quantity_of_each_label(
            objects)

        # convert to format required for dets [x1, y1, x2, y2, confidence]
        dets = [[*start_point, *end_point] for (start_point, end_point, label, confidence) in results]

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(100)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, DETECTION_FRAME_THICKNESS)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    cv2.line(frame, p0, p1, color, 3)

                    if intersect(p0, p1, line[0], line[1]):
                        counter += 1

                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        cv2.line(frame, line[0], line[1], LINE_COLOR, LINE_THICKNESS)

        cv2.putText(frame, str(counter), LINE_COUNTER_POSITION, LINE_COUNTER_FONT, LINE_COUNTER_FONT_SIZE,
                    LINE_COLOR, 2)

        output_movie.write(frame)

        if show:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()