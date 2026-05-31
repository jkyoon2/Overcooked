export const TIMESTEP_MS = 220;
export const PREVIEW_LOOP_MS = 1280;
export const REPLAY_TIMESTEP_MS = 260;
export const DEFAULT_ACTION_IDX = 4;

export const DIRECTION_KEY_CODES = {
    37: 3,
    38: 0,
    39: 2,
    40: 1,
    65: 3,
    68: 2,
    83: 1,
    87: 0,
};

export const INTERACT_KEY_CODES = {
    13: true,
    32: true,
};

export const TERRAIN_SYMBOL_MAP = {
    X: { cssClass: "counter", label: "Counter", frame: "counter.png" },
    O: { cssClass: "onion",   label: "Onion",   frame: "onions.png" },
    T: { cssClass: "tomato",  label: "Tomato",  frame: "tomatoes.png" },
    D: { cssClass: "dish",    label: "Dish",    frame: "dishes.png" },
    P: { cssClass: "pot",     label: "Pot",     frame: "pot.png" },
    S: { cssClass: "serve",   label: "Serve",   frame: "serve.png" },
};
export const DEFAULT_TERRAIN = { cssClass: "floor", label: "", frame: "floor.png" };
