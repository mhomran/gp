# About

The goal of Trackista is to help the managers and coaches by generating the tracking data of players in the key plays of football matches, with an end to end system provided, Trackista has three cameras covering the whole field, then through monitoring the movement of each player, Trackista calculates a top view perspective positions of the players, encapsulating the games state for the key plays of a game into a small feature space that can be used in modelling, and also facilitates conveying managerial instruction for the players, in addition to that Trackista generates insightful statistics about the players with respect to the plays provided, to quantify their movements and evaluate whether it meets expectation.

# System Architecture

This section describes the system architecture for Trackista. The block diagram and the description of each module will be discussed. Our system consists of five main modules which are: Image un-distortion, Image stitching, Model-Field construction, Object Detection, Multi-object tracking and Statistics Generation.

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/System_Architecture.png" alt="System Architecture"></a>
</p>

According to the Block diagram, we have 3 cameras (left, right and center). The first module in the system is Stitching Module which generates stitched image which is used in the whole pipeline.

After that Operator clicks on the playground corners and crossbars to construct model field which discretizes the playground into particles. Then Player detection module takes these particles and tries to search for current particles which cover current player position. After that Player tracking module is responsible for tracking these particles and updating their position using new observations.

Finally, Tracking output is used to generate statistics for each player.

## Image Stitching

After fixing the Radial Distortion of the cameras, the three video feeds are stitched together. 

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/left_center.png" alt="The center and left camera stitched feed"></a>
</p>

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/center_right.png" alt="The center and right camera stitched feed"></a>
</p>

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/left_center_right.png" alt="The final stitched image"></a>
</p>

## Construct a Model-field

Object detection is about surrounding the players by bounding boxes. To get perfect fit bounding boxes, it’s important to find a method that’s different from the common flow of using deep learning models like YOLO which puts a bounding box that can be of any size on the player. Well-fit bounding boxes can be found by utilizing the two-point perspective of the pitch plane. It’s also important to discretize the football pitch into particles for less computation power. The football pitch is also transformed into a top-view football image. Using this transformation, the search of particles whether on the football image or the top view image will become fast with random access.

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/model_field.png" alt="The model field"></a>
</p>


## Player Detection

Player Detection uses the output of model field (particles) to set detections (observations) which are used to correct predictions in the Kalman filter in the tracking module.

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/player_detection_flow.png" alt="Player detection flow"></a>
</p>

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/player_detection_output.png" alt="Player detection output"></a>
</p>


## Player Tracking

The function of tracking module is to get observations of objects and associate these observations with their corresponding tracks; first initialize the tracks with the initial observations, goes on to predict based on the current states, matches these predictions with the incoming stream of observations (detections), this leads keeping the track identity intact with the flow of the analyzed video, tracking data is calculated for the players on the field and are exported to structured files suitable for further analysis.

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/tracking_flow.png" alt="Tracking Flow"></a>
</p>

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/tracking_players.png" alt="Tracking Players"></a>
</p>

## Statistics Generation

This module takes the tracker output and generates statistics for football analysis. This module is important for football analysts because it shows meaningful data for them that will guide them to do their analysis.

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/statistics_flow.png" alt="statistics_flow"></a>
</p>

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/heatmap.png" alt="heatmap"></a>
</p>


