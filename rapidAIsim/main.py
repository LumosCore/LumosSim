import os
import sys
import time
import configparser

from rapidAIsim.core.event.traffic_statistic_event import TrafficStatisticEvent
from rapidAIsim.core.event.change_table_weight_event import ChangeTableWeightEvent
from rapidAIsim.core.event.figret_toe_change_event import FigretToEChangeEvent
from rapidAIsim.core.simulator import Simulator
from rapidAIsim.task.task_generator import Task


def main():
    # pr = cProfile.Profile()
    # pr.enable()

    # Cmd control and instructions
    if len(sys.argv) < 2:
        print('Please add config filename, eg: python3 main.py exp.ini')
        sys.exit(1)
    conf_filename = sys.argv[1]
    conf_handler = configparser.ConfigParser()
    conf_handler.optionxform = lambda option: option  # Enable case sensitive
    conf_handler.read(conf_filename)
    print(f'Load confile: {os.getcwd()}/{conf_filename}')

    Simulator.init_logger()

    Simulator.setup(conf_handler)

    # Create network infrastructure
    Simulator.create_infrastructure()

    # Scheduler
    joint_scheduler = Simulator.CONF_DICT['joint_scheduler']
    print('joint_scheduler =', joint_scheduler, flush=True)
    Simulator.load_scheduler()

    # Generate tasks
    task_obj = Task()
    task_obj.generate()

    # Traffic matrix statistics
    if Simulator.CONF_DICT['traffic_matrix_statistics'] == 'yes': 
        interval = int(Simulator.CONF_DICT['traffic_matrix_statistics_interval'])
        level = Simulator.CONF_DICT['traffic_matrix_level']
        print("Traffic matrix statistics On. Statistic interval is {}s, level is {}".format(interval, level))

        Simulator.reset_traffic_matrix()
        Simulator.register_event(TrafficStatisticEvent(0, interval))

    if Simulator.CONF_DICT['figret_integration'] == 'yes':
        Simulator.reset_traffic_matrix()
        Simulator.register_event(
            ChangeTableWeightEvent(0, int(Simulator.CONF_DICT['figret_te_interval'])))

    if Simulator.CONF_DICT['figret_toe_integration'] == 'yes':
        Simulator.reset_traffic_matrix()
        Simulator.register_event(
            FigretToEChangeEvent(0, int(Simulator.CONF_DICT['figret_toe_interval'])))

    start_time = time.time()
    Simulator.core_run()
    end_time = time.time()
    print("Simulation time: ", end_time - start_time)

    Simulator.close_logger()

    # pr.disable()
    # s = io.StringIO()
    # sortby = "cumtime"
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    # pr.dump_stats("request.prof")


if __name__ == "__main__":
    main()
