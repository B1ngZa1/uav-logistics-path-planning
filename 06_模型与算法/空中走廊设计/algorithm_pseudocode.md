# 空中走廊规划与 MAPPO/图搜索组合伪代码

## 1. 离线建图与层选择

```text
INPUT: building/no-fly map, depots and customers, candidate layers L,
       UAV envelope, d_safe_obs, corridor width, communication/risk fields
OUTPUT: feasible directed corridor graph G=(V,E)

for each geometric candidate edge e=(u,v):
    estimate h_obs[e] and static risk along centerline
    for layer l in L:
        if z_low[l] < h_obs[e] + d_safe_obs: mark (e,l) infeasible
        if intersects no-fly zone or unapproved airspace: mark infeasible
        else:
            compute normalized energy Ebar(e,l)
            compute risk Rbar(e,l), communication cost Cbar(e,l), time Tbar(e,l)
            cost[e,l] = wE*Ebar + wR*Rbar + wC*Cbar + wT*Tbar
    keep all feasible layers with lowest cost and at least one backup layer
    split each edge into lane states q=0..Q[e]-1
    add layer_change_zone and lane_change_zone only outside conflict zones
```

## 2. 单机候选路线

```text
for each UAV i:
    route_set[i] = K_SHORTEST_PATHS(G, start[i], goal[i])
    remove paths violating battery + return reserve constraint
    remove paths with static obstacle clearance < d_safe_obs
```

`K_SHORTEST_PATHS` 可先用 Dijkstra/A*；启发函数使用剩余三维直线距离的下界，不把高度/等待代价漏计到实际代价中。

## 3. 路口预约与冲突消解

```text
INPUT: route_set, predicted speed profile, intersection set J
OUTPUT: time-layer-lane reservation R and conflict-free route choice

priority(i) = (emergency_level[i], urgency[i], -ETA[i], -id[i])
order all requests by priority, then process deterministically

for request (i, route) in order:
    t = earliest_departure(i)
    for every intersection J on route:
        [tin, tout] = predicted_conflict_interval(i, J, t)
        while OVERLAPS([tin,tout], R[J], same_layer_or_insufficient_vertical_sep):
            t = next_available_time(R[J], T_sep[J])
            [tin, tout] = predicted_conflict_interval(i, J, t)
        if same-lane following gap < required_gap:
            t = max(t, predecessor_exit + d_follow/v_i)
        reserve R[J] <- (i, layer, lane, tin, tout)
    commit route and reservations

if no feasible t within horizon:
    add wait edge or call CBS/high-level conflict repair
```

### CBS/优先级修复入口

1. 检测最早冲突 (C=(i,j,J,t))。
2. 分支 A：增加 (i) 的等待/车道/层约束；分支 B：增加 (j) 的对应约束。
3. 低层对单机重新跑 A*，代价含等待、能耗和层切换。
4. 达到无冲突、超时或最大冲突树节点数时停止；超时则退回保守优先级预约并显式报告次优。

## 4. 变道与层切换动作屏蔽

```text
MASK(i, state):
    valid = {hold, speed_up, speed_down, wait}
    if in lane_change_zone and target lane has capacity:
        if predicted TTC and front/back gaps satisfy d_safe_uav:
            if no reservation overlap and communication freshness is valid:
                valid += lane_change(target_lane)
    if in layer_change_zone and vertical clearance + battery are feasible:
        valid += layer_change(target_layer)
    if in conflict_zone or communication degraded:
        remove all lane_change and layer_change actions
    if return-reserve constraint would be violated:
        remove speed_up and route-expanding actions
    return valid
```

## 5. MAPPO 训练循环（CTDE）

```text
initialize shared actor pi_theta(o_i) and centralized critic V_phi(s)
for episode = 1..N:
    reset environment, reservation table, recurrent hidden states
    for t = 0..T-1:
        for each UAV i:
            o_i <- local state + masked lane/layer/route features
            a_i ~ pi_theta(. | o_i)       # high-level request + speed/control
        joint_action <- safety_project(a_1..a_N, reservation table)
        s_next, r, done <- dynamics.step(joint_action)
        store (s, o, a, r, V_phi(s), mask, done)
        update reservation table and route progress
        if done: break
    compute GAE advantages with centralized V_phi(s)
    normalize advantages; update actor with PPO clipping
    update critic with value loss and optional value clipping
    evaluate hard constraints: collision, minimum clearance, illegal action count
```

执行时不把全局状态输入 actor；critic 的全局输入只用于训练。动作 mask 与安全投影在策略外层执行，避免“靠碰撞惩罚学习不可碰撞”。

## 6. MATD3 替换接口

当动作主要是连续 (a_i=[a_x,a_y,a_z,v_{ref}]) 时，可以将 MAPPO actor/critic 换为 MATD3：集中双 Q critic 输入 ((s,a_1,ldots,a_N))，actor 分散执行；目标策略平滑和延迟更新用于抑制 Q 过估计。走廊图、预约表、动作 mask、(1)–(16) 约束保持不变。

## 7. 输出指标

每回合至少输出：完成率、零碰撞率、最小 UAV 净距、最小障碍物净距、非法动作数、平均等待、路口吞吐量、路径长度、能耗、失联时长和返航保留比例。任何“完成率提高”结论必须同时报告硬安全指标。
