/**
 * Module: Tạo Event Callout Box (ĐÃ BỊ VÔ HIỆU HÓA)
 * Lý do: Callout Box dùng 'left %' tuyệt đối bị trôi dạt sang trái mỗi khi realtime append.
 * Các thông tin chi tiết về sự kiện đã được chuyển sang Tooltip (xem tooltip.js)
 */
export function renderAnnotations(containerId, x_data, y_data, states) {
    // Xóa toàn bộ các thẻ DOM cũ nếu có
    const container = document.getElementById(containerId);
    if (container) container.innerHTML = '';
    return;
}