# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the error body and the exceptions that carry it."""

import http

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import errors


class ErrorTest(absltest.TestCase):

  def test_an_error_carries_every_field(self) -> None:
    error = errors.Error(code=errors.ErrorCode.NOT_FOUND, message="No game.")
    self.assertEqual(
        contract.to_json(error),
        {
            "contract_version": contract.CONTRACT_VERSION,
            "code": "not_found",
            "message": "No game.",
            "seat": None,
            "ply": None,
        },
    )

  def test_each_code_maps_to_an_http_status(self) -> None:
    self.assertEqual(
        errors.ErrorCode.BAD_REQUEST.http_status, http.HTTPStatus.BAD_REQUEST
    )
    self.assertEqual(
        errors.ErrorCode.NOT_FOUND.http_status, http.HTTPStatus.NOT_FOUND
    )
    self.assertEqual(
        errors.ErrorCode.METHOD_NOT_ALLOWED.http_status,
        http.HTTPStatus.METHOD_NOT_ALLOWED,
    )
    self.assertEqual(
        errors.ErrorCode.ILLEGAL_ACTION.http_status, http.HTTPStatus.CONFLICT
    )

  def test_an_error_concerns_no_negative_seat_or_ply(self) -> None:
    with self.assertRaisesRegex(ValueError, "seat -1"):
      errors.Error(code=errors.ErrorCode.NOT_FOUND, message="x", seat=-1)
    with self.assertRaisesRegex(ValueError, "ply -1"):
      errors.Error(code=errors.ErrorCode.NOT_FOUND, message="x", ply=-1)


class RequestErrorTest(absltest.TestCase):

  def test_the_base_class_cannot_be_raised(self) -> None:
    with self.assertRaisesRegex(TypeError, "subclass"):
      raise errors.RequestError("x")

  def test_a_subclass_must_declare_a_code(self) -> None:
    with self.assertRaisesRegex(TypeError, "declares no ErrorCode"):
      type("NoCodeError", (errors.RequestError,), {})

  def test_a_controller_failure_names_only_the_seat(self) -> None:
    body = errors.ControllerFailedError(seat=1).to_error()
    self.assertEqual(body.code, errors.ErrorCode.CONTROLLER_FAILED)
    self.assertEqual(body.seat, 1)
    self.assertEqual(
        body.message, "Seat 1's controller failed; undo, reset or change it."
    )

  def test_each_code_has_exactly_one_exception(self) -> None:
    codes = [
        cls.code
        for cls in errors.RequestError.__subclasses__()
        if cls.__module__ == errors.__name__
    ]
    self.assertCountEqual(codes, list(errors.ErrorCode))

  def test_an_exception_becomes_its_error_body(self) -> None:
    error = errors.IllegalActionError("Action 9 is not legal.", seat=0, ply=3)
    self.assertEqual(
        error.to_error(),
        errors.Error(
            code=errors.ErrorCode.ILLEGAL_ACTION,
            message="Action 9 is not legal.",
            seat=0,
            ply=3,
        ),
    )

  def test_the_seat_and_ply_default_to_none(self) -> None:
    body = errors.NotFoundError("There is no session 'x'.").to_error()
    self.assertEqual((body.seat, body.ply), (None, None))


if __name__ == "__main__":
  absltest.main()
