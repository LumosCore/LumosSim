

# Prerequisites
We recommend you to install the packages in `requirements.txt`
and then run `pip install .` in the root directory of this project. `cmake` and cpp build tools are also required.
Please set the path in `CMakeLists.txt`.

Perform a script audit to identify all file paths containing `xxx/`, and update these paths according to user-defined directory locations.

The project also integrates the Figret open-source library, which will subsequently be utilized to evaluate the impact of TE on DCN tasks.

# Get Started

## Simple Test
```
cd large_exp_4096GPU
bash start.sh
```


# Feature
Simulate real time through global static simulator and event base class.

Task generator generate numerous jobs.

Global topology including link capacity.

Jobs can share links.

Update link occupancy at every task event.

Network refresh after every event is done.

Different routing schemes are supported.

Initial configuration templating.

Large-scale verification.

Multi-stage controller.

Ring adn Butterfly strategy.

Multiple tasks can be excuted in turn.

The real start time of every task is determined by TaskStartEvent and TASK_WAITING_LIST.

Automatic Logger.

Measurement event measure link sharing.

Collision weight mechanism.

Support single GPU occupation.
