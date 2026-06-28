import React, { useState, useEffect, useMemo } from "react";
import OvercookedBoard from "../spa/board.jsx";
import { apiPost } from "../utils.js";
import ModalShell from "../components/ModalShell.jsx";
import ChefVisual from "../components/ChefVisual.jsx";
import IngredientStrip from "../components/IngredientStrip.jsx";
import InteractiveTrialRunner from "../components/InteractiveTrialRunner.jsx";

export default function TutorialStage(props) {
    const trialIndex = props.runnerState.trialIndexByStage[props.stage.id] || 0;
    const trialId = props.stage.trial_ids[trialIndex];
    const trial = trialId ? props.session.trials[trialId] : null;
    const [selectedRecipeId, setSelectedRecipeId] = useState((props.initialValues && props.initialValues.team_goal_recipe_id) || props.session.recipe_options[0].id);
    const [savedSetup, setSavedSetup] = useState(Boolean(props.initialValues && props.initialValues.team_goal_recipe_id));
    const [started, setStarted] = useState(false);
    const [briefIndex, setBriefIndex] = useState(-1);
    const [error, setError] = useState("");

    useEffect(() => {
        setStarted(false);
        setBriefIndex(-1);
    }, [trialId]);

    const briefingSteps = useMemo(() => {
        if (!trial) { return []; }
        if (trial.id === "tutorial_solo") {
            return [
                {
                    title: "You are the gray chef",
                    body: "First, practice the kitchen alone without help from the AI chef.",
                    visual: (
                        <div className="role-card__visual">
                            <ChefVisual hatColor="gray" />
                            <span className="ingredient-strip__plus">→</span>
                            <IngredientStrip ingredients={["tomato", "tomato", "tomato"]} />
                        </div>
                    ),
                },
                {
                    title: "Your team goal",
                    body: "The recipe you choose here will be used for the tutorial kitchen.",
                    visual: <IngredientStrip ingredients={(props.session.recipe_options.find((recipe) => recipe.id === selectedRecipeId) || props.session.recipe_options[0]).ingredients} />,
                },
                {
                    title: "Solo practice flow",
                    body: "Bring 3 ingredients to the pot, wait 20 seconds for cooking, pick up the soup with a dish, and deliver it.",
                },
            ];
        }
        return [
            {
                title: "Now cook with the AI chef",
                body: "The orange chef is the AI teammate you will collaborate with and predict during the study.",
                visual: (
                    <div className="role-card__visual">
                        <ChefVisual hatColor="gray" />
                        <span className="ingredient-strip__plus">+</span>
                        <ChefVisual hatColor="orange" />
                    </div>
                ),
            },
            {
                title: "Shared goal",
                body: "Keep working toward the selected recipe together while the AI chef helps in the same kitchen.",
                visual: <IngredientStrip ingredients={(props.session.recipe_options.find((recipe) => recipe.id === selectedRecipeId) || props.session.recipe_options[0]).ingredients} />,
            },
            {
                title: "Probe workflow",
                body: "A practice probe will appear. Choose the AI chef's next subtask, draw the expected route, rate your confidence, then watch the actual behavior.",
            },
        ];
    }, [trial, props.session.recipe_options, selectedRecipeId]);

    function persistSetup() {
        return apiPost("/save_session_section", {
            user_info: props.userInfo,
            section_id: "tutorial_setup",
            data: { team_goal_recipe_id: selectedRecipeId },
        }).then(() => { setSavedSetup(true); });
    }

    function begin() {
        persistSetup().then(() => {
            setBriefIndex(0);
        }).catch((saveError) => {
            setError(String(saveError.message || saveError));
        });
    }

    if (!trial) {
        return (
            <div className="stage-stack">
                <div className="callout">Tutorial complete.</div>
                <div className="stage-actions">
                    <button className="primary-button" type="button" onClick={() => props.onComplete({ tutorial_complete: true, team_goal_recipe_id: selectedRecipeId })}>Continue</button>
                </div>
            </div>
        );
    }

    if (started) {
        return (
            <InteractiveTrialRunner
                trial={trial}
                userInfo={props.userInfo}
                subtaskOptions={props.session.subtask_options}
                onComplete={(summary, trajectory) => {
                    setStarted(false);
                    props.onTrialComplete(trial, summary, trajectory);
                }}
            />
        );
    }

    return (
        <div className="stage-stack">
            <section className="panel-card">
                <div className="role-grid">
                    <div className="role-card role-card--human">
                        <div className="panel-eyebrow">You</div>
                        <div className="role-card__visual"><ChefVisual hatColor="gray" /></div>
                        <h3>Gray chef</h3>
                        <p>You control the gray chef.</p>
                    </div>
                    <div className="role-card role-card--ai">
                        <div className="panel-eyebrow">AI chef</div>
                        <div className="role-card__visual"><ChefVisual hatColor="orange" /></div>
                        <h3>Orange-highlighted chef</h3>
                        <p>The AI chef is the one you observe and predict during probes.</p>
                    </div>
                </div>
                <div className="section-block">
                    <div className="section-block__title">Set the team goal</div>
                    <div className="recipe-grid">
                        {props.session.recipe_options.map((recipe) => (
                            <button
                                className={"recipe-card" + (selectedRecipeId === recipe.id ? " is-selected" : "")}
                                key={recipe.id}
                                type="button"
                                onClick={() => setSelectedRecipeId(recipe.id)}
                            >
                                <div className="recipe-card__visual">
                                    <IngredientStrip ingredients={recipe.ingredients} />
                                </div>
                                <div className="recipe-card__short">{recipe.short_label}</div>
                                <strong>{recipe.label}</strong>
                                <span>{recipe.points} points</span>
                            </button>
                        ))}
                    </div>
                </div>
                <div className="section-block">
                    <div className="section-block__title">How tutorial works</div>
                    <div className="tutorial-steps">
                        <div>1. Practice cooking alone first.</div>
                        <div>2. Practice once more with the AI chef.</div>
                        <div>3. Try the probe workflow before the main session.</div>
                    </div>
                </div>
                {error ? <div className="callout callout--danger">{error}</div> : null}
                <div className="stage-actions">
                    <button className="primary-button" type="button" onClick={begin}>
                        {savedSetup ? "Start next tutorial trial" : "Save goal and start tutorial"}
                    </button>
                </div>
            </section>
            <section className="panel-card">
                <OvercookedBoard
                    layoutGrid={props.session.layout_grid}
                    state={props.previewState}
                    trial={{ human_player_index: 1, target_agent_index: 0, show_target_highlight: false }}
                    label="Tutorial"
                    description="Practice the kitchen before the main session."
                />
            </section>
            <ModalShell open={briefIndex >= 0 && briefIndex < briefingSteps.length} eyebrow="Tutorial" title={briefingSteps[briefIndex] ? briefingSteps[briefIndex].title : "Tutorial"}>
                <div className="panel-stack">
                    {briefingSteps[briefIndex] && briefingSteps[briefIndex].visual ? <div className="guide-visual">{briefingSteps[briefIndex].visual}</div> : null}
                    <p>{briefingSteps[briefIndex] ? briefingSteps[briefIndex].body : ""}</p>
                    <div className="stage-actions">
                        {briefIndex < briefingSteps.length - 1 ? (
                            <button className="primary-button" type="button" onClick={() => setBriefIndex(briefIndex + 1)}>Next</button>
                        ) : (
                            <button className="primary-button" type="button" onClick={() => { setBriefIndex(briefingSteps.length); setStarted(true); }}>Start tutorial</button>
                        )}
                    </div>
                </div>
            </ModalShell>
        </div>
    );
}
