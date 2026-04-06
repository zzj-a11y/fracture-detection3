// 儿童手腕隐匿性骨折检测系统 - 前端对接示例
// 将以下代码添加到完整功能版_6.html中

const API_BASE_URL = 'http://localhost:8000';

// 1. 图像分析上传
async function analyzeImage(file, patientId = null) {
    try {
        const formData = new FormData();
        formData.append('file', file);
        if (patientId) {
            formData.append('patient_id', patientId);
        }

        const response = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }

        return await response.json();
    } catch (error) {
        console.error('图像分析失败:', error);
        return {
            success: false,
            error: error.message,
            message: '分析失败，请检查后端服务是否运行'
        };
    }
}

// 2. 获取患者列表
async function getPatients(search = '', resultFilter = 'all', riskFilter = 'all') {
    try {
        const params = new URLSearchParams();
        if (search) params.append('search', search);
        if (resultFilter !== 'all') params.append('result_filter', resultFilter);
        if (riskFilter !== 'all') params.append('risk_filter', riskFilter);

        const response = await fetch(`${API_BASE_URL}/api/patients?${params}`);
        return await response.json();
    } catch (error) {
        console.error('获取患者列表失败:', error);
        return { success: false, patients: [] };
    }
}

// 3. 创建新患者
async function createPatient(name, age, gender, notes = '') {
    try {
        const params = new URLSearchParams();
        params.append('name', name);
        params.append('age', age);
        params.append('gender', gender);
        if (notes) params.append('notes', notes);

        const response = await fetch(`${API_BASE_URL}/api/patients`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: params
        });

        return await response.json();
    } catch (error) {
        console.error('创建患者失败:', error);
        return { success: false, message: '创建失败' };
    }
}

// 4. 获取统计数据
async function getStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/statistics`);
        return await response.json();
    } catch (error) {
        console.error('获取统计数据失败:', error);
        return { success: false, statistics: {} };
    }
}

// 5. 聊天功能
async function sendChatMessage(message) {
    try {
        const params = new URLSearchParams();
        params.append('message', message);
        params.append('role', 'user');

        const response = await fetch(`${API_BASE_URL}/api/chat/send`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: params
        });

        return await response.json();
    } catch (error) {
        console.error('发送消息失败:', error);
        return { success: false, message: '发送失败' };
    }
}

// 使用示例
async function exampleUsage() {
    console.log('=== 系统API调用示例 ===');

    // 示例1: 检查服务状态
    const statusResponse = await fetch(`${API_BASE_URL}/`);
    console.log('服务状态:', await statusResponse.json());

    // 示例2: 获取患者列表
    const patients = await getPatients();
    console.log('患者列表:', patients);

    // 示例3: 获取统计数据
    const stats = await getStatistics();
    console.log('系统统计:', stats);
}

// 在页面加载完成后运行示例
if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        // 检查后端是否可访问
        fetch(`${API_BASE_URL}/`)
            .then(response => {
                if (response.ok) {
                    console.log('✅ 后端服务正常');
                } else {
                    console.warn('⚠️ 后端服务异常');
                }
            })
            .catch(() => {
                console.error('❌ 无法连接到后端服务');
                console.log('请确保:');
                console.log('1. 后端服务已启动 (python backend.py)');
                console.log('2. 访问地址正确: http://localhost:8000');
                console.log('3. 防火墙允许Python连接');
            });
    });
}

// 导出函数供HTML使用
window.FractureDetectionAPI = {
    analyzeImage,
    getPatients,
    createPatient,
    getStatistics,
    sendChatMessage,
    API_BASE_URL
};