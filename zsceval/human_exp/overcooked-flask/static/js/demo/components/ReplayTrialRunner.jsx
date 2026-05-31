import React, { useState, useEffect, useRef } from "react";
import OvercookedBoard from "../spa/board.jsx";
import ExpectationComposer from "../spa/expectation.jsx";
import { REPLAY_TIMESTEP_MS } from "../constants.js";
import { apiPost, makeTrajId } from "../utils.js";
import ModalShell from "./ModalShell.jsx";
import TrialStats from "./TrialStats.jsx";
import PostTrialRatingModal from "./PostTrialRatingModal.jsx";

export default function ReplayTrialRunner(props) {
    const [trajectory, setTrajectory] = useState(null);
    const [summary, setSummary] = useState(null);
    const [cursor, setCursor] = useState(0);
    const [phase, setPhase] = useState("booting");
    const [activeProbe, setActiveProbe] = useState(null);
    const [probeResponses, setProbeResponses] = useState([]);
    const [probeSubmitted, setProbeSubmitted] = useState(false);
    const [probeDraftKey, setProbeDraftKey] = useState(0);
    const [finishPayload, setFinishPayload] = useState(null);
    const [error, setError] = useState("");
    const finishingRef = useRef(false);

    useEffect(() => {
        let mounted = true;
        apiPost("/start_trial", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
        }).then(() => {
            if (!mounted) { return; }
            if (props.cachedTrajectory) {
                setTrajectory(props.cachedTrajectory.trajectory);
                setSummary(props.cachedTrajectory.summary);
                setPhase("playing");
                return;
            }
            apiPost("/load_trial_trajectory", {
                user_info: props.userInfo,
                trial_id: props.trial.source_trial_id,
            }).then((response) => {
                if (!mounted) { return; }
                setTrajectory(response.trajectory);
                setSummary(response.trial_record ? response.trial_record.trial_summary : null);
                setPhase("playing");
            });
        }).catch((loadError) => {
            if (!mounted) { return; }
            setError(String(loadError.message || loadError));
            setPhase("error");
        });
        return function cleanup() { mounted = false; };
    }, [props.trial.id, props.trial.source_trial_id, props.userInfo, props.cachedTrajectory]);

    function getReplayProbePlan() {
        const observations = trajectory && trajectory.ep_states ? (trajectory.ep_states[0] || []) : [];
        if (!observations.length) { return []; }
        if (summary && summary.probes && summary.probes.length) {
            return summary.probes.slice(0, props.trial.probe.count).map((probeRecord, index) => Object.assign({}, probeRecord, { probe_index: index + 1 }));
        }
        return [1, 2, 3].map((value) => ({
            probe_index: value,
            probe_game_loop: Math.max(1, Math.floor((observations.length * value) / 4)),
        }));
    }

    useEffect(() => {
        if (phase !== "playing" || !trajectory) { return undefined; }
        const replayPlan = getReplayProbePlan();
        const observations = trajectory.ep_states[0] || [];
        const timerId = window.setInterval(() => {
            setCursor((previous) => {
                const nextProbe = replayPlan[probeResponses.length];
                if (nextProbe && previous >= nextProbe.probe_game_loop) {
                    setActiveProbe(nextProbe);
                    setProbeSubmitted(false);
                    setProbeDraftKey((current) => current + 1);
                    setPhase("probe");
                    return previous;
                }
                const next = previous + 1;
                if (next >= observations.length) {
                    window.clearInterval(timerId);
                    setPhase("finishing");
                    return previous;
                }
                return next;
            });
        }, REPLAY_TIMESTEP_MS);
        return function cleanup() { window.clearInterval(timerId); };
    }, [phase, trajectory, probeResponses.length]);

    useEffect(() => {
        if (phase !== "finishing" || finishingRef.current || !summary) { return; }
        finishingRef.current = true;
        const summaryPayload = {
            source_trial_id: props.trial.source_trial_id,
            source_summary: summary,
            replay_expectations: probeResponses,
        };
        apiPost("/save_trial_data", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            updates: summaryPayload,
        }).then(() => apiPost("/finish_episode", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            traj_id: makeTrajId(props.trial.id),
            summary: summaryPayload,
        })).then((payload) => {
            setFinishPayload(payload);
            setPhase("rating");
        }).catch((finishError) => {
            setError(String(finishError.message || finishError));
            setPhase("error");
        });
    }, [phase, props.trial.id, props.trial.source_trial_id, props.userInfo, probeResponses, summary]);

    function submitReplayProbe(expectation) {
        const record = {
            probe_index: activeProbe ? activeProbe.probe_index : probeResponses.length + 1,
            source_probe_game_loop: activeProbe ? activeProbe.probe_game_loop : cursor,
            source_actual_subtask_id: activeProbe ? activeProbe.actual_subtask_id : null,
            source_actual_subtask_label: activeProbe ? activeProbe.actual_subtask_label : null,
            selected_subtask_id: expectation.selectedSubtaskId,
            expected_path: expectation.expectedPath,
            confidence: expectation.confidence,
            response_timestamp: Date.now(),
        };
        setProbeResponses((current) => current.concat([record]));
        setProbeSubmitted(true);
    }

    function resumeAfterReplayProbe() {
        setActiveProbe(null);
        setProbeSubmitted(false);
        setPhase("playing");
    }

    function submitTrialRating(values) {
        const mergedSummary = Object.assign({}, finishPayload.trial_summary, { post_trial_rating: values });
        apiPost("/save_trial_data", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            updates: { post_trial_rating: values, trial_summary: mergedSummary },
        }).then(() => {
            props.onComplete(mergedSummary, null);
        }).catch((ratingError) => {
            setError(String(ratingError.message || ratingError));
            setPhase("error");
        });
    }

    const activeState = trajectory && trajectory.ep_states && trajectory.ep_states[0] ? trajectory.ep_states[0][cursor] : null;

    return (
        <div className="trial-screen">
            <section className="panel-card panel-card--board">
                <div className="trial-hero">
                    <div className="panel-eyebrow">{props.trial.title}</div>
                    <h3>Replay the earlier scene</h3>
                    <p>{props.trial.instruction}</p>
                </div>
                <TrialStats runtime={{
                    score: summary ? summary.score : 0,
                    time_left: trajectory && trajectory.ep_states ? Math.max((trajectory.ep_states[0] || []).length - cursor, 0) : 0,
                    step_count: cursor,
                    probe_records: probeResponses,
                }} />
                <OvercookedBoard
                    layoutGrid={props.trial.layout_grid}
                    state={activeState}
                    trial={Object.assign({}, props.trial, { show_target_highlight: true })}
                />
                {phase === "playing" ? <div className="callout">Watch the replay. It will pause three times and ask what you expected.</div> : null}
                {error ? <div className="callout callout--danger">{error}</div> : null}
            </section>

            <ModalShell open={phase === "probe"} eyebrow="Replay probe" title={probeSubmitted ? "Expectation saved" : "What did you expect here?"}>
                {!probeSubmitted ? (
                    <ExpectationComposer
                        key={"replay-probe-" + probeDraftKey}
                        title="Report what you expected from the AI chef at this moment"
                        probeIndex={activeProbe ? activeProbe.probe_index : probeResponses.length + 1}
                        probeTotal={props.trial.probe.count}
                        prompt={props.trial.probe.prompt}
                        sketchPrompt={props.trial.probe.sketch_prompt}
                        confidencePrompt={props.trial.probe.confidence_prompt}
                        subtaskOptions={props.subtaskOptions}
                        layoutGrid={props.trial.layout_grid}
                        state={activeState}
                        targetAgentIndex={props.trial.target_agent_index}
                        onSubmit={submitReplayProbe}
                    />
                ) : (
                    <div className="panel-stack">
                        <p>Your expectation was saved. Resume replay to compare it with the actual continuation.</p>
                        <div className="stage-actions">
                            <button className="primary-button" type="button" onClick={resumeAfterReplayProbe}>Resume replay</button>
                        </div>
                    </div>
                )}
            </ModalShell>

            <PostTrialRatingModal
                open={phase === "rating" && Boolean(finishPayload)}
                trial={props.trial}
                initialValues={{}}
                onSubmit={submitTrialRating}
            />
        </div>
    );
}
