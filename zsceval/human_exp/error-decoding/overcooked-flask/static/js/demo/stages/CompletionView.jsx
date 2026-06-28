import React from "react";

export default function CompletionView() {
    function exit() {
        try {
            window.close();
        } catch (error) {
            window.location.href = "about:blank";
        }
    }
    return (
        <div className="completion-shell">
            <div className="panel-card completion-card">
                <h2>Session complete</h2>
                <p>Thank you for your participation.</p>
                <p>You have finished the session.</p>
                <p>Please remain seated and wait for the researcher.</p>
                <div className="stage-actions">
                    <button className="primary-button" type="button" onClick={exit}>Exit</button>
                </div>
            </div>
        </div>
    );
}
