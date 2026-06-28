import React, { useState, useMemo } from "react";
import { OBJECTS_ATLAS, OBJECTS_IMAGE_URL } from "../spa/atlas.js";
import { buildTutorialLoopFrames } from "../utils.js";
import ModalShell from "../components/ModalShell.jsx";
import AtlasSprite from "../components/AtlasSprite.jsx";
import LoopingBoardPreview from "../components/LoopingBoardPreview.jsx";
import ObserveHeroPreview from "../components/ObserveHeroPreview.jsx";

export default function OnboardingStage(props) {
    const guideSteps = [
        { id: "move", title: "Move", body: "Use the arrow keys to move your chef." },
        { id: "interact", title: "Interact", body: "Press the space bar to pick up, place, plate, or serve the object (tomato, onion, dish, soup)." },
        { id: "probe", title: "Probe", body: "When a probe appears, choose the AI chef's next subtask, draw the route, and rate your confidence." },
    ];
    const previewSteps = [
        { id: "collect_tomato", label: "Collect tomato" },
        { id: "load_pot", label: "Load pot" },
        { id: "manage_pot", label: "Manage pot" },
        { id: "pickup_soup", label: "Pick up soup" },
        { id: "serve_soup", label: "Serve soup" },
    ];
    const demoFrames = useMemo(() => buildTutorialLoopFrames(), []);
    const probeFrames = useMemo(() => demoFrames.map((frame) => frame.state), [demoFrames]);
    const [guideIndex, setGuideIndex] = useState(-1);
    const activeGuide = guideSteps[guideIndex] || null;

    function openGuide() { setGuideIndex(0); }

    function finishOnboarding() {
        props.onContinue({ reviewed_tutorial: true });
    }

    return (
        <div className="stage-stack">
            <section className="panel-card">
                <div className="panel-stack">
                    <div className="tutorial-preview-shell">
                        <div className="tutorial-preview-shell__header">
                            <div>
                                <div className="panel-eyebrow">Before practice</div>
                                <h3>Watch one full cooking loop</h3>
                            </div>
                            <p>The active subtask lights up as the team moves through the recipe.</p>
                        </div>
                        <LoopingBoardPreview
                            frames={demoFrames}
                            steps={previewSteps}
                            layoutGrid={props.session.layout_grid}
                            trial={{ human_player_index: 1, target_agent_index: 0, show_target_highlight: false }}
                            caption="Gray chef: you. Orange chef: AI teammate."
                        />
                        <div className="key-row">
                            <div className="key-chip">Arrow keys: move</div>
                            <div className="key-chip">Space: interact</div>
                        </div>
                    </div>
                    <div className="stage-actions">
                        <button className="primary-button" type="button" onClick={openGuide}>Next</button>
                    </div>
                </div>
            </section>
            <ModalShell open={guideIndex >= 0 && guideIndex < guideSteps.length} eyebrow="Tutorial" title={activeGuide ? activeGuide.title : "Tutorial"}>
                <div className="panel-stack">
                    <p>{activeGuide ? activeGuide.body : ""}</p>
                    <div className={"guide-visual" + (activeGuide && activeGuide.id === "probe" ? " guide-visual--board" : "")}>
                        {activeGuide && activeGuide.id === "move" ? (
                            <div className="key-demo">
                                <div className="keycap">↑</div>
                                <div className="keycap">←</div>
                                <div className="keycap">↓</div>
                                <div className="keycap">→</div>
                            </div>
                        ) : null}
                        {activeGuide && activeGuide.id === "interact" ? (
                            <div className="guide-visual__interact">
                                <div className="key-demo">
                                    <div className="keycap">Space</div>
                                </div>
                                <div className="ingredient-strip">
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="tomato.png" size={34} className="mini-sprite" />
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="onion.png" size={34} className="mini-sprite" />
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="dish.png" size={34} className="mini-sprite" />
                                    <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="soup-tomato-dish.png" size={34} className="mini-sprite" />
                                </div>
                            </div>
                        ) : null}
                        {activeGuide && activeGuide.id === "probe" ? (
                            <ObserveHeroPreview
                                frames={probeFrames}
                                layoutGrid={props.session.layout_grid}
                                trial={{ human_player_index: 1, target_agent_index: 0, probe: { count: 3, prompt: "What do you expect from the AI chef?", sketch_prompt: "Sketch the route you expected.", confidence_prompt: "How confident are you?" }, show_target_highlight: false }}
                                subtaskOptions={props.session.subtask_options}
                                showProbe
                            />
                        ) : null}
                    </div>
                    <div className="stage-actions">
                        {guideIndex < guideSteps.length - 1 ? (
                            <button className="primary-button" type="button" onClick={() => setGuideIndex(guideIndex + 1)}>Next</button>
                        ) : (
                            <button className="primary-button" type="button" onClick={finishOnboarding}>Next</button>
                        )}
                    </div>
                </div>
            </ModalShell>
        </div>
    );
}
