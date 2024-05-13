# About

The goal of Trackista is to help the managers and coaches by generating the
tracking data of players in the key plays of football matches, with an end to end system
provided, Trackista has three cameras covering the whole field, then through monitoring
the movement of each player, Trackista calculates a top view perspective positions of the
players, encapsulating the games state for the key plays of a game into a small feature
space that can be used in modelling, and also facilitates conveying managerial instruction
for the players, in addition to that Trackista generates insightful statistics about the players with respect to the plays provided, to quantify their movements and evaluate whether it meets expectation.

# System Architecture

This section describes the system architecture for Trackista. The block diagram and the description of each module will be discussed. Our system consists of five main modules which are: Image un-distortion, Image stitching, Model-Field construction, Object Detection, Multi-object tracking and Statistics Generation.

<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/mhomran/gp/raw/master/assets/System_Architecture.png" alt="System Architecture"></a>
</p>

According to the Block diagram, we have 3 cameras (left, right and center). The
first module in the system is Stitching Module which generates stitched image which is
used in the whole pipeline.

After that Operator clicks on the playground corners and crossbars to construct
model field which discretizes the playground into particles. Then Player detection module
takes these particles and tries to search for current particles which cover current player position. After that Player tracking module is responsible for tracking these particles and
updating their position using new observations.

Finally, Tracking output is used to generate statistics for each player.


