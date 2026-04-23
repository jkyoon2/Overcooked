import React, { useEffect, useRef, useState } from "react";

import {
    CHEFS_ATLAS,
    CHEFS_IMAGE_URL,
    OBJECTS_ATLAS,
    OBJECTS_IMAGE_URL,
    TERRAIN_ATLAS,
    TERRAIN_IMAGE_URL,
    getFrame,
    hatColorForPlayer,
    heldObjectSuffix,
    objectFrameName,
    orientationToFrame,
    terrainSymbolToFrame,
} from "./atlas";

const DEFAULT_TILE_SIZE = 76;

function useBoardScale(baseWidth, baseHeight) {
    const wrapRef = useRef(null);
    const [scale, setScale] = useState(1);

    useEffect(() => {
        const node = wrapRef.current;
        if (!node) {
            return undefined;
        }

        function updateScale() {
            const availableWidth = Math.max(node.clientWidth - 24, 240);
            const fallbackHeight = typeof window !== "undefined" ? (window.innerHeight * 0.58) : baseHeight;
            const containerHeight = node.clientHeight > 0 ? node.clientHeight - 24 : fallbackHeight - 24;
            const availableHeight = Math.max(containerHeight, 220);
            setScale(Math.min(1, availableWidth / baseWidth, availableHeight / baseHeight));
        }

        updateScale();
        let resizeObserver = null;
        if (typeof window.ResizeObserver === "function") {
            resizeObserver = new window.ResizeObserver(updateScale);
            resizeObserver.observe(node);
        }
        window.addEventListener("resize", updateScale);
        return function cleanup() {
            window.removeEventListener("resize", updateScale);
            if (resizeObserver) {
                resizeObserver.disconnect();
            }
        };
    }, [baseWidth, baseHeight]);

    return { wrapRef: wrapRef, scale: scale };
}

function terrainAt(layoutGrid, position) {
    if (!Array.isArray(position) || position.length < 2) {
        return " ";
    }
    const x = position[0];
    const y = position[1];
    if (!layoutGrid[y]) {
        return " ";
    }
    return layoutGrid[y][x] || " ";
}

function atlasStyle(atlas, imageUrl, frameName, size) {
    const frame = getFrame(atlas, frameName);
    if (!frame) {
        return {
            width: size,
            height: size,
        };
    }
    const frameDef = frame.frame;
    const scale = size / frameDef.w;
    return {
        width: size,
        height: size,
        backgroundImage: 'url("' + imageUrl + '")',
        backgroundRepeat: "no-repeat",
        backgroundPosition: "-" + String(frameDef.x * scale) + "px -" + String(frameDef.y * scale) + "px",
        backgroundSize: String(atlas.meta.size.w * scale) + "px " + String(atlas.meta.size.h * scale) + "px",
        imageRendering: "pixelated",
    };
}

function SpriteFrame(props) {
    const style = atlasStyle(props.atlas, props.imageUrl, props.frameName, props.size);
    return <div className={props.className} style={style} />;
}

function TileLayer(props) {
    const tileSize = props.tileSize;
    return (
        <div className="eeg-board__terrain">
            {props.layoutGrid.map((row, y) =>
                row.split("").map((symbol, x) => (
                    <div
                        className="eeg-board__tile"
                        key={"tile-" + x + "-" + y}
                        style={{
                            width: tileSize,
                            height: tileSize,
                            left: x * tileSize,
                            top: y * tileSize,
                        }}
                    >
                        <SpriteFrame
                            atlas={TERRAIN_ATLAS}
                            imageUrl={TERRAIN_IMAGE_URL}
                            frameName={terrainSymbolToFrame(symbol)}
                            size={tileSize}
                            className="eeg-board__sprite"
                        />
                    </div>
                ))
            )}
        </div>
    );
}

function ObjectLayer(props) {
    const tileSize = props.tileSize;
    const objects = props.state && props.state.objects ? props.state.objects : [];
    return (
        <div className="eeg-board__objects">
            {objects.map((objectState, index) => {
                const terrainSymbol = terrainAt(props.layoutGrid, objectState.position);
                const frameName = objectFrameName(objectState, terrainSymbol);
                if (!frameName) {
                    return null;
                }
                return (
                    <div
                        className="eeg-board__entity eeg-board__entity--object"
                        key={"object-" + index + "-" + objectState.position.join("-")}
                        style={{
                            width: tileSize,
                            height: tileSize,
                            transform: "translate(" + String(objectState.position[0] * tileSize) + "px, " + String(objectState.position[1] * tileSize) + "px)",
                        }}
                    >
                        <SpriteFrame
                            atlas={OBJECTS_ATLAS}
                            imageUrl={OBJECTS_IMAGE_URL}
                            frameName={frameName}
                            size={tileSize}
                            className="eeg-board__sprite"
                        />
                    </div>
                );
            })}
        </div>
    );
}

function PlayerLayer(props) {
    const tileSize = props.tileSize;
    const players = props.state && props.state.players ? props.state.players : [];
    return (
        <div className="eeg-board__players">
            {players.map((playerState, index) => {
                const direction = orientationToFrame(playerState.orientation);
                const suffix = heldObjectSuffix(playerState);
                const bodyFrame = direction + suffix + ".png";
                const hatFrame = direction + "-" + hatColorForPlayer(index, props.trial) + "hat.png";
                const isHuman = index === props.trial.human_player_index;
                const isTarget = props.trial.show_target_highlight !== false && index === props.trial.target_agent_index;
                return (
                    <div
                        className={
                            "eeg-board__entity eeg-board__entity--player" +
                            (isHuman ? " is-human" : " is-ai") +
                            (isTarget ? " is-target" : "")
                        }
                        key={"player-" + index}
                        style={{
                            width: tileSize,
                            height: tileSize,
                            transform: "translate(" + String(playerState.position[0] * tileSize) + "px, " + String(playerState.position[1] * tileSize) + "px)",
                        }}
                    >
                        <SpriteFrame
                            atlas={CHEFS_ATLAS}
                            imageUrl={CHEFS_IMAGE_URL}
                            frameName={bodyFrame}
                            size={tileSize}
                            className="eeg-board__sprite"
                        />
                        <SpriteFrame
                            atlas={CHEFS_ATLAS}
                            imageUrl={CHEFS_IMAGE_URL}
                            frameName={hatFrame}
                            size={tileSize}
                            className="eeg-board__sprite eeg-board__sprite--overlay"
                        />
                    </div>
                );
            })}
        </div>
    );
}

function PreviewFrame(props) {
    return (
        <div className="eeg-preview">
            <div className="eeg-preview__badge">{props.label}</div>
            <p>{props.description}</p>
        </div>
    );
}

export default function OvercookedBoard(props) {
    const tileSize = props.tileSize || DEFAULT_TILE_SIZE;
    const rows = props.layoutGrid.length;
    const cols = props.layoutGrid[0].length;
    const baseWidth = cols * tileSize;
    const baseHeight = rows * tileSize;
    const boardScale = useBoardScale(baseWidth, baseHeight);
    if (!props.state) {
        return <PreviewFrame label={props.label || "Session"} description={props.description || "Preparing the board."} />;
    }

    return (
        <div className="eeg-board-wrap" ref={boardScale.wrapRef}>
            <div className="eeg-board-stage" style={{ height: baseHeight * boardScale.scale }}>
                <div
                    className="eeg-board"
                    style={{
                        width: baseWidth,
                        height: baseHeight,
                        transform: "scale(" + String(boardScale.scale) + ")",
                    }}
                >
                    <TileLayer layoutGrid={props.layoutGrid} tileSize={tileSize} />
                    <ObjectLayer layoutGrid={props.layoutGrid} state={props.state} tileSize={tileSize} />
                    <PlayerLayer trial={props.trial} state={props.state} tileSize={tileSize} />
                </div>
            </div>
        </div>
    );
}
