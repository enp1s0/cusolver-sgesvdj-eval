#include <cassert>
#include <random>
#include <lapacke.h>
#include <cutf/memory.hpp>
#include <cutf/cusolver.hpp>
#include <mateval/comparison_cuda.hpp>

struct svdj_params {
	float ctol;
};

void lapack_svdj(
		const std::size_t m,
		const std::size_t n,
		const float* const host_A, // M x N
		float* const host_U, // M x N
		float* const host_S, // N
		float* const host_V, // N x N
		const svdj_params params
		) {
#pragma omp parallel for
	for (std::size_t i = 0; i < m * n; i++) {
		host_U[i] = host_A[i];
	}

	const auto lwork = std::max<std::size_t>(6, m + n);
	auto work_uptr = cutf::memory::get_host_unique_ptr<float>(lwork);

	//lapack_int LAPACKE_sgesvj_work( int matrix_layout, char joba, char jobu,
	//                                char jobv, lapack_int m, lapack_int n, float* a,
	//                                lapack_int lda, float* sva, lapack_int mv,
	//                                float* v, lapack_int ldv, float* work,
	//                                lapack_int lwork );
	//lapack_int LAPACKE_sgesvj( int matrix_layout, char joba, char jobu, char jobv,
	//                           lapack_int m, lapack_int n, float* a, lapack_int lda,
	//                           float* sva, lapack_int mv, float* v, lapack_int ldv,
	//                           float* stat );
	work_uptr.get()[0] = params.ctol;

	const auto info = LAPACKE_sgesvj_work(
			LAPACK_COL_MAJOR,
			'G',
			'C', // 'C'?
			'V', // 'A'?
			m, n,
			host_U,
			m,
			host_S,
			n,
			host_V,
			n,
			work_uptr.get(),
			lwork);
}

void cusolver_svdj(
		const std::size_t m,
		const std::size_t n,
		const float* const host_A, // M x N
		float* const host_U, // M x N
		float* const host_S, // N
		float* const host_V, // N x N
		const svdj_params params
		) {
	auto da_uptr = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto du_uptr = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto dv_uptr = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto ds_uptr = cutf::memory::get_device_unique_ptr<float>(n);

	cutf::memory::copy(da_uptr.get(), host_A, m * n);

	gesvdjInfo_t svdj_params;
	const double tol = params.ctol * LAPACKE_slamch('E');
	const unsigned num_svdj_iter = 1000;
	CUTF_CHECK_ERROR(cusolverDnCreateGesvdjInfo(&svdj_params));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetMaxSweeps(svdj_params, num_svdj_iter));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetTolerance(svdj_params, tol));


	auto cusolver_handle_uptr = cutf::cusolver::dn::get_handle_unique_ptr();
	int tmp_working_memory_size;
	CUTF_CHECK_ERROR(cusolverDnSgesvdj_bufferSize(
				*cusolver_handle_uptr.get(),
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				m, n,
				da_uptr.get(), m,
				ds_uptr.get(),
				du_uptr.get(), m,
				dv_uptr.get(), n,
				&tmp_working_memory_size,
				svdj_params
				));

	const auto working_memory_device_size = tmp_working_memory_size;
	auto working_memory_device_uptr = cutf::memory::get_device_unique_ptr<float>(working_memory_device_size);

	auto devInfo_uptr = cutf::memory::get_device_unique_ptr<int>(1);

	CUTF_CHECK_ERROR(cusolverDnSgesvdj(
				*cusolver_handle_uptr.get(),
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				m, n,
				da_uptr.get(), m,
				ds_uptr.get(),
				du_uptr.get(), m,
				dv_uptr.get(), n,
				working_memory_device_uptr.get(),
				working_memory_device_size,
				devInfo_uptr.get(),
				svdj_params
				));

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	cutf::memory::copy(host_U, du_uptr.get(), m * n);
	cutf::memory::copy(host_V, dv_uptr.get(), n * n);
	cutf::memory::copy(host_S, ds_uptr.get(), n);
}

void svdj_eval(
		const std::size_t m,
		const std::size_t n,
		const std::string mode
		) {
	assert(m >= n);

	auto A_uptr = cutf::memory::get_managed_unique_ptr<float>(m * n);
	auto U_uptr = cutf::memory::get_managed_unique_ptr<float>(m * n);
	auto S_uptr = cutf::memory::get_managed_unique_ptr<float>(n);
	auto V_uptr = cutf::memory::get_managed_unique_ptr<float>(n * n);

	std::normal_distribution<float> N_dist(0.f, 1.f);
	std::mt19937 mt(0);
	for (std::size_t i = 0; i < m * n; i++) {
		A_uptr.get()[i] = N_dist(mt);
	}

	svdj_params params{.ctol = 2};

	if (mode == "lapack") {
		lapack_svdj(
				m, n,
				A_uptr.get(),
				U_uptr.get(),
				S_uptr.get(),
				V_uptr.get(),
				params
				);
	} else {
		cusolver_svdj(
				m, n,
				A_uptr.get(),
				U_uptr.get(),
				S_uptr.get(),
				V_uptr.get(),
				params
				);
	}

	const auto residual = mtk::mateval::cuda::residual_UxSxVt(
			m, n, n,
			mtk::mateval::col_major,
			mtk::mateval::col_major,
			mtk::mateval::col_major,
			U_uptr.get(), m,
			S_uptr.get(),
			V_uptr.get(), n,
			A_uptr.get(), m
			);

	const auto orth_U = mtk::mateval::cuda::orthogonality(
			m, n,
			mtk::mateval::col_major,
			U_uptr.get(), m
			);

	const auto orth_V = mtk::mateval::cuda::orthogonality(
			n, n,
			mtk::mateval::col_major,
			V_uptr.get(), n
			);

	std::printf("%s,%lu,%lu,%e,%e,%e\n",
			mode.c_str(),
			m, n,
			residual,
			orth_U,
			orth_V
			);
}

int main() {
	std::printf("mode,m,n,residual,u_orth,v_orth\n");
	for (unsigned N = 256; N <= (1u << 12); N <<= 1) {
		svdj_eval(N, N, "lapack");
		svdj_eval(N, N, "cusolver");
	}
}
