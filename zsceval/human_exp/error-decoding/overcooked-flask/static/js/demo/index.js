import React, { useEffect, useState, useRef } from "react";
import ReactDOM from "react-dom";

import {
    apiPost,
    readJsonStorage,
    writeJsonStorage,
    dedupe,
    buildPreviewState,
    buildRunnerStateFromServer,
} from "./utils.js";

import TopProgress from "./components/TopProgress.jsx";
import StageHeader from "./components/StageHeader.jsx";
import CompletionView from "./stages/CompletionView.jsx";
import WelcomeStage from "./stages/WelcomeStage.jsx";
import ModeOverviewStage from "./stages/ModeOverviewStage.jsx";
import ConsentStage from "./stages/ConsentStage.jsx";
import OnboardingStage from "./stages/OnboardingStage.jsx";
import TutorialStage from "./stages/TutorialStage.jsx";
import StaticStage from "./stages/StaticStage.jsx";
import SurveyStage from "./stages/SurveyStage.jsx";
import TrialBlockStage from "./stages/TrialBlockStage.jsx";

function SessionApp() {
    const storedUserInfo = readJsonStorage("before_game") || {};
    const [bootState, setBootState] = useState({ loading: true, error: "", bundle: null });
    const [userInfo, setUserInfo] = useState(storedUserInfo);
    const [runnerState, setRunnerState] = useState(null);
    const trajectoryCacheRef = useRef({});
    const previewStateRef = useRef(buildPreviewState([
        "XXXXXXXXXXXXX",
        "O   DTXTD   O",
        "XX    P    XX",
        "S   2 P 1   S",
        "XXXXXTXTXXXXX",
    ]));

    function bootstrap(nextUserInfo) {
        setBootState({ loading: true, error: "", bundle: null });
        return apiPost("/session_config", nextUserInfo && nextUserInfo.name ? { user_info: nextUserInfo } : {})
            .then((bundle) => {
                setBootState({ loading: false, error: "", bundle: bundle });
                setRunnerState(buildRunnerStateFromServer(bundle.session, bundle, nextUserInfo || {}));
            })
            .catch((error) => {
                setBootState({ loading: false, error: String(error.message || error), bundle: null });
            });
    }

    useEffect(() => {
        bootstrap(storedUserInfo);
    }, []);

    useEffect(() => {
        if (runnerState) {
            writeJsonStorage("eeg_runner_state_spa", runnerState);
        }
    }, [runnerState]);

    function persistProgress(nextRunnerState, currentStageId, payload, actorOverride) {
        const actor = actorOverride || userInfo;
        if (!actor || !actor.name) {
            return Promise.resolve();
        }
        return apiPost("/save_session_section", {
            user_info: actor,
            section_id: "client_progress",
            data: {
                runner_state: nextRunnerState,
                payload: payload || {},
            },
            progress: {
                current_stage_id: currentStageId,
                completed_stage_ids: nextRunnerState.completedStages,
            },
        });
    }

    function completeStage(stage, data, userOverride, baseRunnerState) {
        const runner = baseRunnerState || runnerState;
        const actor = userOverride || userInfo;
        const completedStages = dedupe((runner.completedStages || []).concat([stage.id]));
        const nextStageIndex = Math.min(runner.stageIndex + 1, bootState.bundle.session.stages.length);
        const nextStage = bootState.bundle.session.stages[nextStageIndex] || null;
        const nextRunnerState = {
            stageIndex: nextStageIndex,
            trialIndexByStage: Object.assign({}, runner.trialIndexByStage),
            completedStages: completedStages,
            completedTrials: Object.assign({}, runner.completedTrials),
            trialSummaries: Object.assign({}, runner.trialSummaries),
            sectionResponses: Object.assign({}, runner.sectionResponses),
        };
        nextRunnerState.sectionResponses[stage.id] = data || {};

        const savePromise = actor && actor.name ? apiPost("/save_session_section", {
            user_info: actor,
            section_id: stage.id,
            data: data || {},
            progress: {
                current_stage_id: nextStage ? nextStage.id : "session_complete",
                completed_stage_ids: completedStages,
            },
        }) : Promise.resolve();

        return savePromise.then(() => {
            setRunnerState(nextRunnerState);
            if (actor && actor.name) {
                return persistProgress(nextRunnerState, nextStage ? nextStage.id : "session_complete", data || {}, actor);
            }
            return null;
        });
    }

    function handleIntakeSubmit(values) {
        setUserInfo(values);
        return completeStage(bootState.bundle.session.stages[runnerState.stageIndex], values, values);
    }

    function handleSurveySubmit(stage, values) {
        completeStage(stage, values);
    }

    function handleTrialComplete(trial, summary, trajectory) {
        const stage = bootState.bundle.session.stages[runnerState.stageIndex];
        const currentCount = runnerState.trialIndexByStage[stage.id] || 0;
        const nextCount = currentCount + 1;
        const nextTrialIndexByStage = Object.assign({}, runnerState.trialIndexByStage);
        nextTrialIndexByStage[stage.id] = nextCount;
        const nextCompletedTrials = Object.assign({}, runnerState.completedTrials);
        nextCompletedTrials[trial.id] = true;
        const nextTrialSummaries = Object.assign({}, runnerState.trialSummaries);
        nextTrialSummaries[trial.id] = summary;
        if (trajectory) {
            trajectoryCacheRef.current[trial.id] = { trajectory: trajectory, summary: summary };
        }

        const updatedRunnerState = {
            stageIndex: runnerState.stageIndex,
            trialIndexByStage: nextTrialIndexByStage,
            completedStages: runnerState.completedStages.slice(),
            completedTrials: nextCompletedTrials,
            trialSummaries: nextTrialSummaries,
            sectionResponses: Object.assign({}, runnerState.sectionResponses),
        };

        const blockFinished = nextCount >= stage.trial_ids.length;
        if (blockFinished) {
            setRunnerState(updatedRunnerState);
            completeStage(stage, { block_complete: true, last_trial_id: trial.id }, null, updatedRunnerState);
            return;
        }

        setRunnerState(updatedRunnerState);
        persistProgress(updatedRunnerState, stage.id, { current_trial_id: trial.id });
    }

    if (bootState.loading || !runnerState) {
        return (
            <div className="loading-shell">
                <div className="loading-card">
                    <div className="loader" />
                    <h2>Loading session...</h2>
                </div>
            </div>
        );
    }

    if (bootState.error) {
        return (
            <div className="loading-shell">
                <div className="loading-card loading-card--error">
                    <h2>Failed to load session</h2>
                    <p>{bootState.error}</p>
                </div>
            </div>
        );
    }

    const session = bootState.bundle.session;
    const stage = session.stages[runnerState.stageIndex] || null;
    if (!stage) {
        return <CompletionView />;
    }

    let stageBody = null;
    if (stage.type === "welcome") {
        stageBody = <WelcomeStage session={session} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "mode_overview") {
        stageBody = <ModeOverviewStage session={session} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "intake_form") {
        stageBody = <ConsentStage intakeForm={bootState.bundle.intake_form} userInfo={userInfo} onSubmit={handleIntakeSubmit} />;
    } else if (stage.type === "onboarding") {
        stageBody = <OnboardingStage stage={stage} session={session} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "tutorial_lab") {
        stageBody = (
            <TutorialStage
                stage={stage}
                session={session}
                userInfo={userInfo}
                previewState={previewStateRef.current}
                runnerState={runnerState}
                initialValues={runnerState.sectionResponses.tutorial_setup || runnerState.sectionResponses[stage.id] || {}}
                onTrialComplete={handleTrialComplete}
                onComplete={(payload) => completeStage(stage, payload)}
            />
        );
    } else if (stage.type === "static") {
        stageBody = <StaticStage stage={stage} onContinue={(payload) => completeStage(stage, payload)} />;
    } else if (stage.type === "survey") {
        stageBody = <SurveyStage stage={stage} initialValues={runnerState.sectionResponses[stage.id]} onSubmit={(values) => handleSurveySubmit(stage, values)} />;
    } else if (stage.type === "trial_block") {
        stageBody = (
            <TrialBlockStage
                stage={stage}
                session={session}
                userInfo={userInfo}
                runnerState={runnerState}
                previewState={previewStateRef.current}
                trajectoryCache={trajectoryCacheRef.current}
                onTrialComplete={handleTrialComplete}
                onAdvanceEmptyBlock={() => completeStage(stage, { block_complete: true })}
            />
        );
    }

    return (
        <div className="app-shell">
            <TopProgress session={session} runnerState={runnerState} />
            <main className="app-main">
                <StageHeader stage={stage} session={session} runnerState={runnerState} />
                {stageBody}
            </main>
        </div>
    );
}

ReactDOM.render(<SessionApp />, document.getElementById("app-root"));
