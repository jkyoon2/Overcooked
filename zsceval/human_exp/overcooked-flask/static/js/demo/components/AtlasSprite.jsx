import React from "react";
import { atlasStyle } from "../utils.js";

export default function AtlasSprite(props) {
    return (
        <div
            className={props.className || "mini-sprite"}
            style={atlasStyle(props.atlas, props.imageUrl, props.frameName, props.size || 40)}
        />
    );
}
