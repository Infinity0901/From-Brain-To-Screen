# Virtual Reality EEG

The user is able to move in the virtual world just thinking about moving in a specific direction.

Material required : Muse 2, Oculus Quest 2

1. Install librairies : 
    => pip install tensorflow

    => pip install numpy

    => pip install pylsl 

    => pip install keyboard
    
2. Lauch the stream in a new window : muselsl stream
3. Launch the visualisation in an other window to make sur the Muse 2 is well placed : muselsl view ou muselsl view --version 2
4. Clone this repository
5. In Unity : 
    => Edit => Project Settings : Install XR Plugin Management

        => Initialize XR on Startup on computer and android by clicking on OpenXR

        => Under XR Plugin Management click on Open XR : 

            => On Computer add these Interaction Profiles : HTC Vive Controller / Microsoft Motion Controller Profile / Oculus Touch Controller Profile / Valve Index Controller Profile
            
            => On Android add this Interaction Profile : Oculus Touch Controller Profile 
    
    => Click on play in Unity
