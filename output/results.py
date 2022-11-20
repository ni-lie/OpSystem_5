from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import re
from typing import Sequence, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns


N = 30
PIDS = (3, 4, 5)
POLICIES = ('baseline', 'double', 'biased')
POLICIES_VS_BASELINE = ('double', 'biased')
METRICS = (
    # 'first_execution_tick',
    # 'arrival_tick',
    # 'ending_tick',
    'context_switches',
    'response_time',
    'total_running_time',
    'turnaround_time',
    'percentage_of_cpu_time',
)


class State(IntEnum):
    UNUSED = 0
    EMBRYO = 1
    SLEEPING = 2
    RUNNABLE = 3
    RUNNING = 4
    ZOMBIE = 5


@dataclass(frozen=True)
class Process:
    pid: int
    name: str
    state: State

    @classmethod
    def from_str(cls, s: str) -> Process:
        match = re.search(
            r'\[(?P<pid>\d+)\](\*|\s)?(?P<name>[^:]+):(?P<state>\d+)', s)

        if match:
            pid = int(match.group('pid'))
            name = match.group('name')
            state = State(int(match.group('state')))

            return Process(
                pid=pid,
                name=name,
                state=state,
            )

        else:
            raise RuntimeError(f'Failed to parse string as process: [{s}]')


@dataclass(frozen=True)
class ProcessTable:
    mapping: dict[int, Process]

    @classmethod
    def from_list(cls, processes: Sequence[Process]) -> ProcessTable:
        return ProcessTable(
            mapping={process.pid: process for process in processes},
        )

    def is_pid_present(self, pid: int) -> bool:
        return pid in self.mapping

    def get_pid_state(self, pid: int) -> State | None:
        if pid not in self.mapping:
            return None

        return self.mapping[pid].state


@dataclass(frozen=True)
class TickState:
    tick: int
    processes: ProcessTable

    @classmethod
    def from_line(cls, line: str) -> TickState:
        tokens = [token.strip() for token in line.split('|')]

        tick = int(tokens[0])
        processes: Sequence[Process] = []

        for process_str in tokens[1:]:
            processes.append(Process.from_str(process_str))

        return TickState(
            tick=tick,
            processes=ProcessTable.from_list(processes),
        )

    def is_pid_present(self, pid: int) -> bool:
        return self.processes.is_pid_present(pid)


class ProcessMetrics(TypedDict):
    pid: int
    policy: str
    run: int
    first_execution_tick: int
    arrival_tick: int
    ending_tick: int
    response_time: int
    turnaround_time: int
    percentage_of_cpu_time: float
    total_running_time: int
    context_switches: int


class MetricStats(TypedDict):
    pid: int
    policy: str
    metric: str
    mean: float
    median: float
    range: float
    sd: float


@dataclass(frozen=True)
class TickLog:
    tick_states: Sequence[TickState]

    @classmethod
    def from_lines(cls, lines: Sequence[str]) -> TickLog:
        return TickLog(
            tick_states=sorted([TickState.from_line(line.strip())
                               for line in lines], key=lambda s: s.tick),
        )

    def get_ticks_with_pid(self, pid: int) -> Sequence[TickState]:
        return [tick_state for tick_state in self.tick_states if tick_state.is_pid_present(pid)]

    def get_first_tick_state(self) -> TickState:
        return self.tick_states[0]

    def get_last_tick_state(self) -> TickState:
        return self.tick_states[-1]

    def get_tick_state_pairs(self) -> Sequence[tuple[TickState, TickState]]:
        return list(zip(self.tick_states, self.tick_states[1:]))


class MetricComputer:
    def get_arrival_tick(self, tick_log: TickLog, pid: int) -> int:
        for tick_state in tick_log.get_ticks_with_pid(pid):
            if tick_state.processes.get_pid_state(pid) in (State.RUNNABLE, State.RUNNING):
                return tick_state.tick

        raise RuntimeError(f'PID {pid} has no arrival tick')

    def get_first_execution_tick(self, tick_log: TickLog, pid: int) -> int:
        for tick_state in tick_log.get_ticks_with_pid(pid):
            if tick_state.processes.get_pid_state(pid) == State.RUNNING:
                return tick_state.tick

        raise RuntimeError(f'PID {pid} has no first execution tick')

    def get_ending_tick(self, tick_log: TickLog, pid: int) -> int:
        for tick_state in tick_log.get_ticks_with_pid(pid):
            if tick_state.processes.get_pid_state(pid) == State.ZOMBIE:
                return tick_state.tick

        raise RuntimeError(f'PID {pid} has no ending tick')

    def get_response_time(self, tick_log: TickLog, pid: int) -> int:
        return self.get_first_execution_tick(tick_log, pid) - self.get_arrival_tick(tick_log, pid)

    def get_turnaround_time(self, tick_log: TickLog, pid: int) -> int:
        return self.get_ending_tick(tick_log, pid) - self.get_arrival_tick(tick_log, pid)

    def get_cpu_first_tick(self, tick_log: TickLog) -> int:
        tick_state = tick_log.get_first_tick_state()

        if tick_state:
            return tick_state.tick

        raise RuntimeError(f'Log has no first tick')

    def get_cpu_last_tick(self, tick_log: TickLog) -> int:
        tick_state = tick_log.get_last_tick_state()

        if tick_state:
            return tick_state.tick

        raise RuntimeError(f'Log has no last tick')

    def get_running_times(self, tick_log: TickLog, pid: int) -> Sequence[int]:
        running_times: Sequence[int] = []

        for (cur_tick_state, next_tick_state) in tick_log.get_tick_state_pairs():
            if cur_tick_state.processes.get_pid_state(pid) != State.RUNNING:
                continue

            start_tick = cur_tick_state.tick
            end_tick = next_tick_state.tick

            running_times.append(end_tick - start_tick)

        return running_times

    def get_total_running_time(self, tick_log: TickLog, pid: int) -> int:
        return sum(self.get_running_times(tick_log, pid))

    def get_percentage_of_cpu_time(self, tick_log: TickLog, pid: int) -> float:
        first_tick = tick_log.get_first_tick_state().tick
        last_tick = tick_log.get_last_tick_state().tick

        duration = last_tick - first_tick
        total_running_time = self.get_total_running_time(tick_log, pid)

        return total_running_time / duration

    def get_context_switches(self, tick_log: TickLog, pid: int) -> int:
        return sum([1 for tick_states in tick_log.get_ticks_with_pid(pid)
                    if tick_states.processes.get_pid_state(pid) == State.RUNNING])

    def get_process_metrics(self, tick_log: TickLog, pid: int, policy: str, run: int) -> ProcessMetrics:
        return {
            'pid': pid,
            'policy': policy,
            'run': run,
            'first_execution_tick': self.get_first_execution_tick(tick_log, pid),
            'arrival_tick': self.get_arrival_tick(tick_log, pid),
            'response_time': self.get_response_time(tick_log, pid),
            'ending_tick': self.get_ending_tick(tick_log, pid),
            'turnaround_time': self.get_turnaround_time(tick_log, pid),
            'percentage_of_cpu_time': self.get_percentage_of_cpu_time(tick_log, pid),
            'total_running_time': self.get_total_running_time(tick_log, pid),
            'context_switches': self.get_context_switches(tick_log, pid),
        }


def get_data_stats(df: pd.DataFrame):
    stats: list[MetricStats] = []

    for pid in (3, 4, 5):
        for policy in POLICIES:
            for metric in METRICS:
                filtered = df[(df['pid'] == pid) & (  # pyright: ignore
                    df['policy'] == policy)]

                mean = filtered[metric].mean()  # pyright: ignore
                median = filtered[metric].median()  # pyright: ignore

                data_min = min(filtered[metric])  # pyright: ignore
                data_max = max(filtered[metric])  # pyright: ignore
                data_range = data_max - data_min  # pyright: ignore

                sd = filtered[metric].std()  # pyright: ignore

                stats.append({
                    'pid': pid,
                    'policy': policy,
                    'metric': metric,
                    'mean': mean,
                    'median': median,
                    'range': data_range,
                    'sd': sd,
                })

    return pd.DataFrame(stats)


def plot(df: pd.DataFrame):
    sns.set_theme(style="ticks", palette="pastel")  # pyright: ignore

    for metric in METRICS:
        sns.catplot(  # pyright: ignore
            data=df, kind="bar",
            x="policy", y=metric, hue="pid",
            errorbar="sd", palette="dark", alpha=.6, height=6
        )
        plt.show()  # pyright: ignore

        sns.boxplot(x="policy", y=metric, hue="pid", data=df)  # pyright: ignore
        plt.show()  # pyright: ignore


def get_relevant_lines(lines: Sequence[str]) -> Sequence[str]:
    lines = [line for line in lines
             if line.strip() != ''
             and not line.startswith('qemu')
             and 'SeaBIOS' not in line
             and not line.startswith('iPXE')
             and not line.startswith('Press ')
             and not line.startswith('Booting ')
             and not line.startswith('cpu0:')
             and not line.startswith('sb:')
             and not line.startswith('init:')
             and not line.startswith('xv6...')
             and not line.startswith('$ ')
             and "[3]*test:4" not in line
             and "[4]*test:4" not in line
             and "[5]*test:4" not in line]

    last_idx = None

    for idx, line in list(enumerate(lines))[::-1]:
        if '[2]*test:4' in line:
            last_idx = idx
            break

    if last_idx is not None:
        lines = lines[:last_idx+1]

    return lines


def main():
    metrics_list: list[ProcessMetrics] = []

    for policy in POLICIES:
        for n in range(1, N + 1):
            with open(f'{policy}-{n}.txt', 'r') as f:
                lines = get_relevant_lines(f.readlines())

            tick_log = TickLog.from_lines(lines)

            for pid in PIDS:
                metrics = MetricComputer().get_process_metrics(tick_log, pid, policy, n)
                metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list)
    print(df.to_string())  # pyright: ignore

    stats = get_data_stats(df)
    print(stats)

    for metric in METRICS:
        for policy in POLICIES_VS_BASELINE:
            for pid in PIDS:
                x = df[(df['policy'] == 'baseline') & (df['pid'] == pid)][metric]  # pyright: ignore
                y = df[(df['policy'] == policy) & (df['pid'] == pid)][metric]  # pyright: ignore

                print(f'{metric}, PID {pid}, baseline vs. {policy}:\t\t{scipy.stats.mannwhitneyu(x, y, alternative="two-sided")}')

    plot(df)


if __name__ == "__main__":
    main()
