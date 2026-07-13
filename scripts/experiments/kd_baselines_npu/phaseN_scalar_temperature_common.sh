#!/usr/bin/env bash

# Accept completion only when the latest log session has the registered Phase N
# configuration and reaches its exact iteration budget. Older completion markers
# cannot satisfy a newer, partial session.
phaseN_log_session_complete() {
  local log_file="$1"
  local target_iterations="$2"
  local expected_kd_temperature="$3"
  local expected_save_per_iters="$4"
  local expected_val_per_iters="$5"
  local expected_skip_val="$6"
  local expected_student_pretrained_base="$7"

  [[ -f "$log_file" ]] || return 1

  awk \
    -v target="$target_iterations" \
    -v kd_temp="$expected_kd_temperature" \
    -v save_every="$expected_save_per_iters" \
    -v val_every="$expected_val_per_iters" \
    -v skip_val="$expected_skip_val" \
    -v imagenet_base="$expected_student_pretrained_base" '
      function reset_session() {
        saw_namespace = 0
        config_ok = 0
        saw_iter = 0
        last_iter = 0
        reached_target = 0
        saw_total_time = 0
        validation_count = 0
        last_validation_iter = 0
        bad_health = 0
      }

      BEGIN { sq = sprintf("%c", 39); reset_session() }

      /Namespace\(/ {
        reset_session()
        saw_namespace = 1
        config_ok = 1
        config_ok = config_ok && index($0, "teacher_model=" sq "deeplabv3" sq ",") > 0
        config_ok = config_ok && index($0, "student_model=" sq "deeplabv3_mobilenet_ssseg" sq ",") > 0
        config_ok = config_ok && index($0, "student_backbone=" sq "mobilenetv3_small" sq ",") > 0
        config_ok = config_ok && index($0, "teacher_backbone=" sq "resnet101" sq ",") > 0
        config_ok = config_ok && index($0, "dataset=" sq "voc" sq ",") > 0
        config_ok = config_ok && index($0, "crop_size=[512, 512],") > 0
        config_ok = config_ok && index($0, "workers=8,") > 0
        config_ok = config_ok && index($0, "batch_size=16,") > 0
        config_ok = config_ok && index($0, "max_iterations=" target ",") > 0
        config_ok = config_ok && index($0, "lr=0.02,") > 0
        config_ok = config_ok && index($0, "momentum=0.9,") > 0
        config_ok = config_ok && index($0, "weight_decay=0.0001,") > 0
        config_ok = config_ok && index($0, "kd_temperature=" kd_temp ",") > 0
        config_ok = config_ok && index($0, "lambda_kd=1.0,") > 0
        config_ok = config_ok && index($0, "lambda_adv=0.001,") > 0
        config_ok = config_ok && index($0, "lambda_d=0.1,") > 0
        config_ok = config_ok && index($0, "lambda_cwd_fea=50.0,") > 0
        config_ok = config_ok && index($0, "lambda_cwd_logit=3.0,") > 0
        config_ok = config_ok && index($0, "use_covar=False,") > 0
        config_ok = config_ok && index($0, "teacher_output_temp=3.0,") > 0
        config_ok = config_ok && index($0, "device_type=" sq "npu" sq ",") > 0
        config_ok = config_ok && index($0, "seed=1234,") > 0
        config_ok = config_ok && index($0, "resume=None,") > 0
        config_ok = config_ok && index($0, "save_per_iters=" save_every ",") > 0
        config_ok = config_ok && index($0, "val_per_iters=" val_every ",") > 0
        config_ok = config_ok && index($0, "student_pretrained_base=" sq imagenet_base sq ",") > 0
        config_ok = config_ok && index($0, "student_pretrained=" sq "None" sq ",") > 0
        config_ok = config_ok && index($0, "skip_val=" skip_val ",") > 0
        next
      }

      /Iters:[[:space:]]*[0-9]+[[:space:]]*\/[[:space:]]*[0-9]+/ {
        iter_text = $0
        sub(/^.*Iters:[[:space:]]*/, "", iter_text)
        split(iter_text, fields, "/")
        iteration = fields[1] + 0
        maximum = fields[2] + 0
        if (saw_iter && iteration < last_iter) {
          reset_session()
        }
        saw_iter = 1
        last_iter = iteration
        if (iteration == target && maximum == target) {
          reached_target = 1
        }
      }

      /Sample:[[:space:]]*1449[[:space:]]*,/ {
        if (saw_namespace) {
          validation_count++
          last_validation_iter = last_iter
        }
      }

      {
        lower_line = tolower($0)
        if (saw_namespace && lower_line ~ /(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)/) bad_health = 1
      }

      /Total training time:/ {
        if (reached_target) {
          saw_total_time = 1
        }
      }

      END {
        validation_ok = (skip_val == "True") ? (validation_count == 0) : (validation_count == int(target / val_every) && last_validation_iter == target)
        if (ENVIRON["PHASEN_STRICT_DEBUG"] == "1") {
          print "saw_namespace=" saw_namespace, "config_ok=" config_ok, "saw_iter=" saw_iter, "last_iter=" last_iter, "reached_target=" reached_target, "saw_total_time=" saw_total_time, "validation_count=" validation_count, "last_validation_iter=" last_validation_iter, "bad_health=" bad_health > "/dev/stderr"
        }
        exit !(saw_namespace && config_ok && saw_iter && reached_target && saw_total_time && validation_ok && !bad_health)
      }
    ' "$log_file"
}

phaseN_artifacts_complete() {
  local save_dir="$1"
  local require_best="$2"
  local model_name="kd_deeplabv3_mobilenet_ssseg_mobilenetv3_small_voc"

  [[ -f "$save_dir/${model_name}.pth" ]] || return 1
  [[ -f "$save_dir/training_state_latest.pth" ]] || return 1

  if [[ "$require_best" == "True" ]]; then
    [[ -f "$save_dir/${model_name}_best_model.pth" ]] || return 1
    [[ -f "$save_dir/training_state_best.pth" ]] || return 1
  fi
}
