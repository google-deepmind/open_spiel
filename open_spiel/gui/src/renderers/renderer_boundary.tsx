// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {Component, type ErrorInfo, type ReactNode} from 'react';

import './renderers.css';

interface BoundaryProps {
  readonly children: ReactNode;
  /** What is drawn under the error instead of the renderer. */
  readonly fallback: ReactNode;
  /** A change of this clears a failure, so the renderer is tried again. */
  readonly resetKey?: string;
}

interface BoundaryState {
  readonly error: Error | null;
}

/**
 * Catches a renderer that throws, most often on a view it cannot read, and
 * shows the error above the fallback, so a broken renderer is loud but the
 * game stays playable from its text and the legal list.
 */
export class RendererBoundary extends Component<BoundaryProps, BoundaryState> {
  override state: BoundaryState = {error: null};

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return {error: error instanceof Error ? error : new Error(String(error))};
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('Renderer failed', error, info.componentStack);
  }

  override componentDidUpdate(previous: BoundaryProps): void {
    if (
      this.state.error !== null &&
      previous.resetKey !== this.props.resetKey
    ) {
      this.setState({error: null});
    }
  }

  override render(): ReactNode {
    const {error} = this.state;
    if (error === null) return this.props.children;
    return (
      <>
        <p className="renderer__error" role="alert">
          {`The board could not be drawn: ${error.message}`}
        </p>
        {this.props.fallback}
      </>
    );
  }
}
