import React, { useEffect, useMemo, useRef, useState } from "react";

import {
    CHEFS_ATLAS,
    CHEFS_IMAGE_URL,
    OBJECTS_ATLAS,
    OBJECTS_IMAGE_URL,
    TERRAIN_ATLAS,
    TERRAIN_IMAGE_URL,
    getFrame,
    terrainSymbolToFrame,
} from "./atlas";

const VISUAL_SIZE = 56;
const DEFAULT_SKETCH_TILE_SIZE = 38;

function atlasStyle(atlas, imageUrl, frameName, size, opacity) {
    const frame = getFrame(atlas, frameName);
    if (!frame) {
        return {
            width: size,
            height: size,
            opacity: opacity === undefined ? 1 : opacity,
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
        opacity: opacity === undefined ? 1 : opacity,
        flex: "0 0 auto",
    };
}

function Sprite(props) {
    return <div className={props.className} style={atlasStyle(props.atlas, props.imageUrl, props.frameName, props.size, props.opacity)} />;
}

function ChefSprite(props) {
    const suffix = props.heldSuffix || "";
    return (
        <div className="subtask-visual__chef">
            <Sprite atlas={CHEFS_ATLAS} imageUrl={CHEFS_IMAGE_URL} frameName={"SOUTH" + suffix + ".png"} size={VISUAL_SIZE} className="subtask-visual__sprite" />
            <Sprite atlas={CHEFS_ATLAS} imageUrl={CHEFS_IMAGE_URL} frameName="SOUTH-orangehat.png" size={VISUAL_SIZE} className="subtask-visual__sprite subtask-visual__sprite--overlay" />
        </div>
    );
}

function ObjectSprite(props) {
    return <Sprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName={props.frameName} size={VISUAL_SIZE} className="subtask-visual__sprite" />;
}

function TerrainSprite(props) {
    return <Sprite atlas={TERRAIN_ATLAS} imageUrl={TERRAIN_IMAGE_URL} frameName={props.frameName} size={VISUAL_SIZE} className="subtask-visual__sprite" />;
}

function SubtaskVisual(props) {
    if (props.subtaskId === "collect_tomato") {
        return <div className="subtask-visual"><ChefSprite /><div className="subtask-visual__arrow">→</div><ObjectSprite frameName="tomato.png" /></div>;
    }
    if (props.subtaskId === "collect_onion") {
        return <div className="subtask-visual"><ChefSprite /><div className="subtask-visual__arrow">→</div><ObjectSprite frameName="onion.png" /></div>;
    }
    if (props.subtaskId === "collect_dish") {
        return <div className="subtask-visual"><ChefSprite /><div className="subtask-visual__arrow">→</div><ObjectSprite frameName="dish.png" /></div>;
    }
    if (props.subtaskId === "load_pot") {
        return <div className="subtask-visual"><ChefSprite heldSuffix="-tomato" /><div className="subtask-visual__arrow">→</div><TerrainSprite frameName="pot.png" /></div>;
    }
    if (props.subtaskId === "manage_pot") {
        return <div className="subtask-visual"><TerrainSprite frameName="pot.png" /><div className="subtask-visual__arrow">↺</div><ObjectSprite frameName="soup-tomato-3-cooking.png" /></div>;
    }
    if (props.subtaskId === "pickup_soup") {
        return <div className="subtask-visual"><ObjectSprite frameName="dish.png" /><div className="subtask-visual__arrow">→</div><ObjectSprite frameName="soup-tomato-dish.png" /></div>;
    }
    if (props.subtaskId === "serve_soup") {
        return <div className="subtask-visual"><ObjectSprite frameName="soup-tomato-dish.png" /><div className="subtask-visual__arrow">→</div><TerrainSprite frameName="serve.png" /></div>;
    }
    if (props.subtaskId === "counter_transfer") {
        return <div className="subtask-visual"><ObjectSprite frameName="tomato.png" /><div className="subtask-visual__arrow">→</div><TerrainSprite frameName="counter.png" /></div>;
    }
    if (props.subtaskId === "wait") {
        return <div className="subtask-visual"><ChefSprite /><div className="subtask-visual__pause">II</div></div>;
    }
    return <div className="subtask-visual"><ChefSprite /><div className="subtask-visual__arrow">⇢</div><TerrainSprite frameName="floor.png" /></div>;
}

function manhattanDistance(pointA, pointB) {
    return Math.abs(pointA[0] - pointB[0]) + Math.abs(pointA[1] - pointB[1]);
}

function appendPoint(points, point, startPosition) {
    const anchor = points.length ? points[points.length - 1] : startPosition;
    if (!anchor) {
        return points;
    }
    if (anchor[0] === point[0] && anchor[1] === point[1]) {
        return points;
    }
    if (manhattanDistance(anchor, point) !== 1) {
        return points;
    }
    return points.concat([point]);
}

function LikertBar(props) {
    const values = [];
    for (let value = props.min; value <= props.max; value += 1) {
        values.push(value);
    }
    return (
        <div className="likert-bar">
            <div className="likert-bar__row">
                {values.map((value) => (
                    <button
                        className={"likert-bar__tick" + (Number(props.value) === value ? " is-selected" : "")}
                        key={value}
                        type="button"
                        onClick={() => props.onChange(value)}
                    >
                        <span>{value}</span>
                    </button>
                ))}
            </div>
            <div className="likert-bar__legend">
                <span>{props.leftLabel}</span>
                <span>{props.rightLabel}</span>
            </div>
        </div>
    );
}

function SketchTile(props) {
    const tileSize = props.tileSize || DEFAULT_SKETCH_TILE_SIZE;
    return (
        <button
            className="trajectory-sketcher__tile"
            type="button"
            style={{
                width: tileSize,
                height: tileSize,
                left: props.x * tileSize,
                top: props.y * tileSize,
            }}
            onMouseDown={props.onPointerStart}
            onMouseEnter={props.onPointerEnter}
            onClick={props.onPointerStart}
        >
            <Sprite
                atlas={TERRAIN_ATLAS}
                imageUrl={TERRAIN_IMAGE_URL}
                frameName={terrainSymbolToFrame(props.symbol)}
                size={tileSize}
                className="trajectory-sketcher__terrain"
                opacity={0.28}
            />
        </button>
    );
}

function TrajectorySketcher(props) {
    const drawingRef = useRef(false);
    const latestPathRef = useRef(props.pathPoints || []);
    const tileSize = props.tileSize || DEFAULT_SKETCH_TILE_SIZE;
    const cols = props.layoutGrid[0].length;
    const rows = props.layoutGrid.length;
    const width = cols * tileSize;
    const height = rows * tileSize;
    const points = useMemo(() => {
        const base = props.startPosition ? [props.startPosition] : [];
        return base.concat(props.pathPoints || []);
    }, [props.pathPoints, props.startPosition]);

    useEffect(() => {
        latestPathRef.current = props.pathPoints || [];
    }, [props.pathPoints]);

    useEffect(() => {
        function finishDraw() {
            drawingRef.current = false;
        }
        window.addEventListener("mouseup", finishDraw);
        return function cleanup() {
            window.removeEventListener("mouseup", finishDraw);
        };
    }, []);

    function handlePointer(point) {
        const nextPoints = appendPoint(latestPathRef.current || [], point, props.startPosition);
        if (nextPoints.length === (latestPathRef.current || []).length) {
            return;
        }
        latestPathRef.current = nextPoints;
        props.onChangePath(nextPoints);
    }

    return (
        <div className="trajectory-sketcher">
            <div className="trajectory-sketcher__canvas" style={{ width: width, height: height }}>
                {props.layoutGrid.map((row, y) =>
                    row.split("").map((symbol, x) => (
                        <SketchTile
                            key={"sketch-" + x + "-" + y}
                            x={x}
                            y={y}
                            symbol={symbol}
                            tileSize={tileSize}
                            onPointerStart={() => {
                                drawingRef.current = true;
                                handlePointer([x, y]);
                            }}
                            onPointerEnter={() => {
                                if (drawingRef.current) {
                                    handlePointer([x, y]);
                                }
                            }}
                        />
                    ))
                )}
                <svg className="trajectory-sketcher__path" viewBox={"0 0 " + String(width) + " " + String(height)}>
                    {points.length > 0 ? (
                        <polyline
                            points={points.map((point) => String(point[0] * tileSize + (tileSize / 2)) + "," + String(point[1] * tileSize + (tileSize / 2))).join(" ")}
                            fill="none"
                            stroke="rgba(61, 217, 178, 0.98)"
                            strokeWidth="5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    ) : null}
                    {points.map((point, index) => (
                        <circle
                            key={"node-" + index}
                            cx={point[0] * tileSize + (tileSize / 2)}
                            cy={point[1] * tileSize + (tileSize / 2)}
                            r={index === 0 ? 8 : 5}
                            fill={index === 0 ? "rgba(100, 168, 255, 1)" : "rgba(61, 217, 178, 1)"}
                        />
                    ))}
                </svg>
            </div>
            <div className="trajectory-sketcher__controls">
                <button className="ghost-button" type="button" onClick={props.onUndo} disabled={!props.pathPoints.length}>Undo</button>
                <button className="ghost-button" type="button" onClick={props.onClear} disabled={!props.pathPoints.length}>Clear</button>
            </div>
            <div className="trajectory-sketcher__note">
                Draw one continuous route. Diagonal moves are not allowed.
            </div>
        </div>
    );
}

function WizardDots(props) {
    return (
        <div className="wizard-dots">
            {props.steps.map((step, index) => (
                <div className={"wizard-dots__dot" + (index === props.currentIndex ? " is-current" : "") + (index < props.currentIndex ? " is-complete" : "")} key={step.id} />
            ))}
        </div>
    );
}

export default function ExpectationComposer(props) {
    const [pageIndex, setPageIndex] = useState(0);
    const [selectedSubtaskId, setSelectedSubtaskId] = useState(props.selectedSubtaskId || "");
    const [pathPoints, setPathPoints] = useState(props.pathPoints || []);
    const [confidence, setConfidence] = useState(props.confidence || "");
    const sketchTileSize = props.sketchTileSize || DEFAULT_SKETCH_TILE_SIZE;
    const visibleSubtaskOptions = props.previewSubtaskLimit
        ? props.subtaskOptions.slice(0, props.previewSubtaskLimit)
        : props.subtaskOptions;
    const targetPlayer = props.state && props.state.players ? props.state.players[props.targetAgentIndex] : null;
    const startPosition = targetPlayer ? targetPlayer.position : null;
    const pages = [
        { id: "subtask", title: "Choose the next subtask" },
        { id: "path", title: "Draw the route you expect" },
        { id: "confidence", title: "Rate your confidence" },
    ];

    useEffect(() => {
        setPageIndex(0);
        setSelectedSubtaskId(props.selectedSubtaskId || "");
        setPathPoints(props.pathPoints || []);
        setConfidence(props.confidence || "");
    }, [props.probeIndex, props.selectedSubtaskId, props.pathPoints, props.confidence]);

    useEffect(() => {
        if (props.forcedPageIndex !== undefined && props.forcedPageIndex !== null) {
            setPageIndex(props.forcedPageIndex);
        }
    }, [props.forcedPageIndex]);

    function canAdvance() {
        if (pageIndex === 0) {
            return Boolean(selectedSubtaskId);
        }
        if (pageIndex === 1) {
            if (selectedSubtaskId === "wait") {
                return true;
            }
            return pathPoints.length > 0;
        }
        return String(confidence).trim().length > 0;
    }

    function submit() {
        props.onSubmit({
            selectedSubtaskId: selectedSubtaskId,
            expectedPath: pathPoints,
            startPosition: startPosition,
            confidence: confidence,
        });
    }

    return (
        <div className={"expectation-probe" + (props.compact ? " expectation-probe--compact" : "")}>
            <div className="expectation-probe__header">
                <div>
                    <div className="panel-eyebrow">Expectation Probe</div>
                    <h3>{props.title}</h3>
                </div>
                <div className="status-chip">Probe {props.probeIndex} / {props.probeTotal}</div>
            </div>
            <p>{props.prompt}</p>
            <WizardDots steps={pages} currentIndex={pageIndex} />
            <section className="expectation-panel">
                <div className="expectation-panel__title">{pages[pageIndex].title}</div>
                {pageIndex === 0 ? (
                    <div className="subtask-grid subtask-grid--visual">
                        {visibleSubtaskOptions.map((option) => (
                            <button
                                className={"subtask-card subtask-card--visual" + (selectedSubtaskId === option.id ? " is-selected" : "")}
                                key={option.id}
                                type="button"
                                onClick={() => setSelectedSubtaskId(option.id)}
                            >
                                <SubtaskVisual subtaskId={option.id} />
                                <div className="subtask-card__meta">
                                    <strong>{option.label}</strong>
                                    {!props.compact ? <span>{option.description}</span> : null}
                                </div>
                            </button>
                        ))}
                    </div>
                ) : null}
                {pageIndex === 1 ? (
                    <div className="panel-stack">
                        <p>{props.sketchPrompt}</p>
                        <TrajectorySketcher
                            layoutGrid={props.layoutGrid}
                            startPosition={startPosition}
                            tileSize={sketchTileSize}
                            pathPoints={pathPoints}
                            onChangePath={setPathPoints}
                            onUndo={() => setPathPoints(pathPoints.slice(0, -1))}
                            onClear={() => setPathPoints([])}
                        />
                    </div>
                ) : null}
                {pageIndex === 2 ? (
                    <div className="panel-stack">
                        <p>{props.confidencePrompt}</p>
                        <LikertBar
                            min={1}
                            max={7}
                            value={confidence}
                            leftLabel="Low confidence"
                            rightLabel="High confidence"
                            onChange={setConfidence}
                        />
                    </div>
                ) : null}
            </section>
            <div className="expectation-probe__actions">
                {!props.readOnly ? (
                    <div className="expectation-probe__button-group">
                        {pageIndex > 0 ? <button className="ghost-button" type="button" onClick={() => setPageIndex(pageIndex - 1)}>Back</button> : null}
                        {pageIndex < pages.length - 1 ? (
                            <button className="primary-button" type="button" disabled={!canAdvance()} onClick={() => setPageIndex(pageIndex + 1)}>Next</button>
                        ) : (
                            <button className="primary-button" type="button" disabled={!canAdvance()} onClick={submit}>Save Expectation</button>
                        )}
                    </div>
                ) : (
                    <button className="primary-button" type="button" disabled>{pageIndex < pages.length - 1 ? "Next" : "Save Expectation"}</button>
                )}
            </div>
        </div>
    );
}
