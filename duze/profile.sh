nvprof --export-profile timeline_c11.nvprof ./gpu_main_c11 < test/autotest_04a.in > out.out
#nvprof --kernels compute_gig_wt_kernel --metrics all -o metrics_c11.nvprof ./gpu_main_c11 < test/autotest_04a.in > out.out
#nvprof --kernels compute_gig_wt_kernel --events all -o events_c11.nvprof ./gpu_main_c11 < test/autotest_04a.in > out.out
nvprof --kernels compute_gig_wt_kernel --analysis-metrics -o a_metrics_c11.nvprof ./gpu_main_c11 < test/autotest_04a.in > out.out
