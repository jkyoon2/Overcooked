#!/bin/bash
# Kill all train_agent_pool.sh / agent_pool_sp related processes

LAYOUT=$1

if [ -z "$LAYOUT" ]; then
    echo "Killing all agent_pool_sp processes..." >&2
    # 1. Kill parent bash scripts first to prevent spawning new processes
    pkill -9 -f "train_agent_pool.sh" 2>/dev/null
    # 2. Kill Python train processes launched for agent pool
    pkill -9 -f "train/train_sp.py.*agent_pool_sp" 2>/dev/null
    # 3. Kill worker processes (setproctitle names)
    pkill -9 -f "mappo-Overcooked.*-agent_pool_sp" 2>/dev/null
    pkill -9 -f "rmappo-Overcooked.*-agent_pool_sp" 2>/dev/null
else
    echo "Killing agent_pool_sp processes for layout: ${LAYOUT}..." >&2
    # 1. Kill parent bash scripts first to prevent spawning new processes
    pkill -9 -f "train_agent_pool.sh.*${LAYOUT}" 2>/dev/null
    # 2. Kill Python train processes launched for this layout
    pkill -9 -f "train/train_sp.py.*agent_pool_sp.*${LAYOUT}" 2>/dev/null
    pkill -9 -f "train/train_sp.py.*${LAYOUT}.*agent_pool_sp" 2>/dev/null
    pkill -9 -f "train/train_sp.py.*layout_name ${LAYOUT}.*agent_pool_sp" 2>/dev/null
    # 3. Kill worker processes (setproctitle names)
    pkill -9 -f "mappo-Overcooked.*${LAYOUT}.*-agent_pool_sp" 2>/dev/null
    pkill -9 -f "rmappo-Overcooked.*${LAYOUT}.*-agent_pool_sp" 2>/dev/null
fi

sleep 1

# Check remaining
REMAINING_PARENT=$(ps aux | grep "train_agent_pool.sh" | grep -v grep 2>/dev/null)
REMAINING_TRAIN=$(ps aux | grep "train/train_sp.py" | grep "agent_pool_sp" | grep -v grep 2>/dev/null)
REMAINING_WORKERS=$(ps aux | grep "Overcooked.*agent_pool_sp" | grep -v grep 2>/dev/null)

echo "" >&2
if [ -z "$REMAINING_PARENT" ] && [ -z "$REMAINING_TRAIN" ] && [ -z "$REMAINING_WORKERS" ]; then
    echo "[OK] All processes killed" >&2
else
    if [ ! -z "$REMAINING_PARENT" ]; then
        echo "[WARN] Remaining parent scripts:" >&2
        echo "$REMAINING_PARENT" >&2
    fi
    if [ ! -z "$REMAINING_TRAIN" ]; then
        echo "[WARN] Remaining train_sp(agent_pool_sp):" >&2
        echo "$REMAINING_TRAIN" >&2
    fi
    if [ ! -z "$REMAINING_WORKERS" ]; then
        echo "[WARN] Remaining workers:" >&2
        echo "$REMAINING_WORKERS" >&2
    fi
fi
