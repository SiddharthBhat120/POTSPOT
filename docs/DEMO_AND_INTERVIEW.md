# PotSpot: demo script and interview notes

## Demo video script (about 2 minutes)

Record the screen at 1080p with the browser zoomed to 100%. Speak over it; numbers below are from this repo's own runs.

| Time | On screen | Say |
|---|---|---|
| 0:00 | A phone on a handlebar, then the empty PotSpot map | "City corporations find potholes one complaint at a time. My phone already sees every pothole on my commute. PotSpot turns that video into a map." |
| 0:12 | Click **Add a ride**, pick the video and GPX, keep Automatic sync, upload | "I record the road with the phone facing forward and a free GPS logger running. PotSpot lines up the two clocks itself by matching how fast the road moves in the video to the GPS speed." |
| 0:30 | Progress bar in the ride card, then dots appear along the route | "It samples 15 frames a second and runs a YOLO11 model I fine-tuned. A tracker follows each pothole across frames, so a pothole seen in 14 frames is one dot, not 14." |
| 0:50 | Open `annotated.mp4` from the trip folder, pause on a box with its track number | "Here is what the tracker sees: the same ID stays on the pothole as it slides down the frame. Anything seen in fewer than 3 frames is dropped as a flicker." |
| 1:05 | Click a red dot, the detail panel opens | "Each pothole gets its best photo, with people and number plates blurred, a width estimate from where it sits in the frame, and a Google Maps link." |
| 1:20 | Toggle the heatmap, then filter to Large | "Ride the same road again and the sightings merge within 8 metres, so the map shows which potholes are still there." |
| 1:35 | Click **Download report for the city**, scroll the PDF | "And this is the report you send to the city: the worst potholes grouped by road, with photos and coordinates." |
| 1:50 | README results table | "On 155 held-out test photos it reaches 0.62 mAP50. The README lists where it fails: shadows, wet patches and manhole covers." |

## Resume bullets

- Built PotSpot, a computer-vision pipeline (fine-tuned YOLO11 + ByteTrack + GPS sync) that maps potholes from phone dashcam video; reached **0.62 mAP50** on a held-out test set of 155 photos, with recall **0.86** on close potholes.
- Cut **1,238 raw detections to 22 map records with zero double counts** on a 42 s ride with known ground truth, by adding tracklet stitching to ByteTrack after diagnosing lost tracks on a perspective-correct synthetic clip; processes video in real time on a CPU.
- Found that **57% of the dataset photos had near-duplicate twins** and split by perceptual-hash group to avoid an inflated test score; shipped a FastAPI + React/Leaflet app with automatic video-GPS clock sync, cross-ride merging, privacy blurring and a PDF report for the city, covered by 22 tests.

Once you have mapped your own rides, swap in the real-road line from the spec: "mapped **N potholes over K km** of real roads".

## Interview questions

**1. What are precision, recall and mAP, and which matters more here?**
Precision is the share of reported potholes that are real; recall is the share of real potholes the model finds. mAP50 summarises the whole precision-recall curve: average precision when a predicted box counts as correct if its IoU with the labelled box is at least 0.5 (mAP50-95 averages IoU thresholds from 0.5 to 0.95, so it also rewards tight boxes). Here recall matters more. A missed pothole stays on the road; a false one costs an inspector a minute, and false ones are filtered again by the 3-frame rule and by repeat rides. So I run the detector at a low confidence (0.2) and let tracking remove noise. On the test set: mAP50 0.621, precision 0.72 and recall 0.54, and recall rises to 0.86 for large (close) potholes.

**2. Why do you need tracking? What breaks without it?**
At 15 frames a second a pothole stays in view for one to three seconds, so it is detected 15 to 45 times. Without tracking every detection is a dot, the count is 30 times too high, and "potholes per km" means nothing. Tracking gives each pothole one ID across frames, so I keep one record with its best frame. On the demo ride the detector fired 1,238 times; tracking turned that into 22 records with no pothole counted twice.

**3. ByteTrack lost close potholes. How did you find and fix that?**
I built a synthetic road clip with perspective-correct potholes at known positions and sizes, so I knew exactly how many tracks there should be. At 10 fps a pothole near the bottom of the frame moves further between frames than its own height, IoU with the predicted box drops to zero, and ByteTrack starts a new ID. Two fixes: sample at 15 fps by default, and a stitching step that joins a new track to an ended one if it starts within 0.6 s, near the old track's velocity-predicted position, and with a box at least half as big (potholes only grow as you approach).

**4. How do you sync video time and GPS time? What if they drift?**
Everything is stored as "the video started offset seconds after the first GPX point". There are three ways to get the offset: the user types it, the video's creation_time metadata, or automatic: I measure optical-flow speed in the lower half of the frame, which is roughly proportional to ground speed, and slide it against the GPS speed curve to find the offset with the highest correlation. Stops and accelerations make the curve distinctive. The offset is constant per ride, so clock drift on long rides slowly shifts positions; the fix is to split long rides, or fit offset plus a drift rate (a line instead of a constant) with the same correlation search over windows.

**5. Where on the road does a pothole get placed?**
At the GPS position when it was last seen, which is when it is closest to the camera, moved 4 m ahead along the heading because the camera looks ahead of the bike. Using the first sighting would put it 15 to 20 m too early.

**6. How is severity estimated, and how much do you trust it?**
With a pinhole camera over a flat road, a point at image row y below the horizon is at distance h·f/(y − horizon), and a width of w pixels at that distance is w·h/(y − horizon) metres. So width ≈ camera height × box width ÷ (box bottom − horizon row), no focal length needed. On a synthetic clip with known sizes it measured 0.28 m for 0.3 m and 0.81 m for 0.9 m. But it assumes a flat road and the right mount height and horizon; a slope or a tilted phone moves the horizon and the estimate. It cannot see depth at all. I present it as small / medium / large and say "estimate" everywhere.

**7. How do you merge sightings from different rides?**
Greedy clustering on distance: a new sighting joins the nearest existing pothole within 8 m (about phone GPS accuracy), measured to the cluster's centroid so chains of points cannot drift into one giant cluster. Sightings from the same ride never merge with each other, because the tracker already decided they are different potholes. The merged record keeps the worst severity and counts sightings and rides.

**8. Where does the model fail, and how would you fix it?**
I looked at the worst misses and the most confident false alarms. Misses: water-filled potholes reflecting the sky, shallow worn patches, potholes under shadows, and far ones only a few pixels high (recall 0.33 for small boxes vs 0.86 for large). False alarms: tar seal lines, skid marks and sandy patches, plus many "errors" that are really label disagreements, one box around a cluster versus three separate boxes. Fixes: add hard negatives (shadows, patches, manhole covers labelled as background), more night and rain images such as RDD2022, a bigger model (yolo11s at 640 px on a GPU), and use the tracker's evidence: a real pothole grows smoothly as you approach, a shadow often does not.

**9. Your test score could be inflated. How did you check?**
57% of the dataset photos have a near-identical twin: frames a moment apart or re-encoded copies. A random split puts twins on both sides of the split, and the model is tested on photos it has nearly seen. I grouped photos with a perceptual hash (dHash, threshold picked by looking at pairs) and kept each group in one split. Augmented rain, blur and dusk copies are made only from training photos.

**10. How would you run this on the phone itself?**
Export the model to TFLite or Core ML with int8 quantisation (Ultralytics exports both), run it at 5 to 10 fps on the phone's NPU at 320 to 416 px, run the tracker on-device (ByteTrack is cheap, just Kalman filters and IoU matching), and read GPS directly so clock sync disappears. Upload only the small blurred crops and coordinates, not the video, which saves data and is better for privacy. The server then only does merging, road names and reports.
