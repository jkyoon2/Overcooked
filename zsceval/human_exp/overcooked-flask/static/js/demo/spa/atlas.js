const chefsAtlas = require("../../../assets/chefs.json");
const objectsAtlas = require("../../../assets/objects.json");
const terrainAtlas = require("../../../assets/terrain.json");

export const CHEFS_ATLAS = chefsAtlas;
export const OBJECTS_ATLAS = objectsAtlas;
export const TERRAIN_ATLAS = terrainAtlas;

export const CHEFS_IMAGE_URL = "/static/assets/chefs.png";
export const OBJECTS_IMAGE_URL = "/static/assets/objects.png";
export const TERRAIN_IMAGE_URL = "/static/assets/terrain.png";

export function orientationToFrame(orientation) {
    const key = Array.isArray(orientation) ? orientation.join(",") : String(orientation);
    switch (key) {
        case "0,-1":
            return "NORTH";
        case "0,1":
            return "SOUTH";
        case "1,0":
            return "EAST";
        case "-1,0":
            return "WEST";
        default:
            return "SOUTH";
    }
}

export function terrainSymbolToFrame(symbol) {
    switch (symbol) {
        case "X":
            return "counter.png";
        case "O":
            return "onions.png";
        case "T":
            return "tomatoes.png";
        case "D":
            return "dishes.png";
        case "P":
            return "pot.png";
        case "S":
            return "serve.png";
        default:
            return "floor.png";
    }
}

export function getFrame(atlas, frameName) {
    return atlas.frames[frameName] || null;
}

function inferSoupProfile(objectState) {
    const ingredients = objectState && objectState._ingredients ? objectState._ingredients : [];
    const names = ingredients.map((ingredient) => ingredient.name);
    const tomatoCount = names.filter((name) => name === "tomato").length;
    const onionCount = names.filter((name) => name === "onion").length;
    return tomatoCount >= onionCount ? "tomato" : "onion";
}

function inferSoupCount(objectState) {
    const ingredients = objectState && objectState._ingredients ? objectState._ingredients : [];
    return ingredients.length || 0;
}

export function soupFrameName(objectState, terrainSymbol, heldByPlayer) {
    const profile = inferSoupProfile(objectState);
    const count = inferSoupCount(objectState);
    if (heldByPlayer) {
        return "soup-" + profile;
    }
    if (terrainSymbol !== "P") {
        return "soup-" + profile + "-dish.png";
    }
    if (objectState.is_ready) {
        return "soup-" + profile + "-cooked.png";
    }
    if (count >= 3 && !objectState.is_cooking) {
        return "soup-" + profile + "-3-0.png";
    }
    if (count >= 3) {
        return "soup-" + profile + "-3-cooking.png";
    }
    return "soup-" + profile + "-" + String(Math.max(count, 1)) + "-cooking.png";
}

export function objectFrameName(objectState, terrainSymbol) {
    if (!objectState) {
        return null;
    }
    switch (objectState.name) {
        case "onion":
            return "onion.png";
        case "tomato":
            return "tomato.png";
        case "dish":
            return "dish.png";
        case "soup":
            return soupFrameName(objectState, terrainSymbol, false);
        default:
            return null;
    }
}

export function heldObjectSuffix(playerState) {
    const heldObject = playerState && playerState.held_object ? playerState.held_object : null;
    if (!heldObject) {
        return "";
    }
    switch (heldObject.name) {
        case "onion":
            return "-onion";
        case "tomato":
            return "-tomato";
        case "dish":
            return "-dish";
        case "soup":
            return "-soup-" + inferSoupProfile(heldObject);
        default:
            return "";
    }
}

export function hatColorForPlayer(playerIndex, trial) {
    if (playerIndex === trial.human_player_index) {
        return "gray";
    }
    if (playerIndex === trial.target_agent_index) {
        return "orange";
    }
    return "blue";
}
