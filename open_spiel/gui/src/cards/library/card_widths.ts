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

// The readable card-width range every fitted layout clamps to (px). Pure, so
// core/ can read it without importing the React layout primitives.
/** The narrowest card a fitted layout draws, in px. */
export const MIN_CARD_WIDTH = 40;
/** The widest card a fitted layout draws, in px. */
export const MAX_CARD_WIDTH = 110;
